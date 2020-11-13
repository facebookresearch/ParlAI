#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import queue
import random
from typing import List, Any, Dict, Tuple, Set, TYPE_CHECKING

from mephisto.core.logger_core import get_logger
from mephisto.data_model.blueprint import SharedTaskState, TaskRunner
from omegaconf import DictConfig

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun
    from mephisto.data_model.assignment import Unit
    from mephisto.data_model.agent import Agent

logger = get_logger(name=__name__, verbose=True, level="info")

DEFAULT_TASK_CONFIG = {
    "hit_title": "Which Conversational Partner is Better?",
    "hit_description": "Evaluate quality of conversations through comparison.",
    "hit_keywords": "chat,evaluation,comparison,conversation",
}


PairingsDict = Dict[str, Any]
WorkerID = str
UnitID = str


# TODO(#99) ask the run to enqueue new tasks when running out and still
# unfinished tasks remain.
class AcuteEvalRunner(TaskRunner):
    """
    Managing class for the acute evaluator process.

    Relevant args are parsed in the `setup_args` function above.
    """

    def __init__(
        self, task_run: "TaskRun", args: DictConfig, shared_state: SharedTaskState
    ):
        """
        Initialize the AcuteEvaluator.

        The following object attributes are used in running ACUTE Eval:

        ``onboarding_tasks``: A list of ALL available _onboarding_ comparison tasks

        ``desired_tasks``: A list of ALL available comparison tasks

        ``task_queue``: A queue of REMAINING tasks, from which HITs are constructed.

        ``worker_data``: A mapping from worker ID to data about the worker, including
        their tasks completed, conversations seen, and onboarding todo

        ``failed_onboard``: The set of workers who have failed onboarding

        ``unit_agent_map``: Map from unit id to the worker_id and task data for cleanup
        """
        super().__init__(task_run, args, shared_state)
        random.seed(self.args.blueprint.random_seed)
        self.is_concurrent = False
        self.assignment_duration_in_seconds = (
            task_run.get_task_config().assignment_duration_in_seconds
        )

        # class attributes
        self.onboarding_tasks: List[Dict] = []
        self.desired_tasks: List[Dict] = []
        self.task_queue: queue.Queue = queue.Queue()
        self.worker_data: Dict[WorkerID, Dict[str, List]] = {}
        self.failed_onboard: Set = set()
        self.unit_agent_map: Dict[UnitID, Tuple[WorkerID, List[PairingsDict]]] = {}

        # read in conversations data
        self._load_conversation_data()

        # setup the task queue
        self._setup_task_queue()

    def _get_worker_data(self, worker_id: str) -> Dict[str, List]:
        """
        Return worker data if present, else a default dict.
        """
        onboarding_todo = list(range(len(self.onboarding_tasks)))
        random.shuffle(onboarding_todo)
        self.worker_data[worker_id] = self.worker_data.get(
            worker_id,
            {
                "tasks_completed": [],
                "conversations_seen": [],
                "onboarding_todo": onboarding_todo,
            },
        )
        return self.worker_data[worker_id]

    def set_block_qual(self, task_id: str):
        """
        Set block qualification if necessary.

        :param task_id:
            task id used to set block qualification, if necessary.
        """
        if self.args.blueprint.block_on_onboarding_fail:
            self.block_qualification = self.args.blueprint.block_qualification
            if self.block_qualification is None:
                self.block_qualification = f"{task_id}_failed_onboarding"
                self.args.blueprint.block_qualification = self.block_qualification
                logger.warning(
                    "No block_qualification set in opt, automatically creating "
                    "new qualification {}".format(self.block_qualification)
                )
            found_qualifications = self.task_run.db.find_qualifications(
                self.block_qualification
            )
            if len(found_qualifications) == 0:
                self.task_run.db.make_qualification(self.block_qualification)

    def _load_conversation_data(self):
        """
        Load conversation data.

        Loads in the data from the pairs filepath.
        """
        pairs_path = self.args.blueprint.pairings_filepath

        with open(pairs_path) as pf:
            for i, l in enumerate(pf.readlines()):
                convo_pair = json.loads(l.strip())
                eval_speakers = [
                    s
                    for d in convo_pair["dialogue_dicts"]
                    for s in d["speakers"]
                    if s in convo_pair["speakers_to_eval"]
                ]
                # make sure order is preserved
                assert eval_speakers == convo_pair["speakers_to_eval"]
                model_left_idx = random.choice([0, 1])
                task = {
                    "task_specs": {
                        "s1_choice": self.args.blueprint.s1_choice,
                        "s2_choice": self.args.blueprint.s2_choice,
                        "question": self.args.blueprint.eval_question,
                        "is_onboarding": convo_pair["is_onboarding"],
                        "model_left": {
                            "name": eval_speakers[model_left_idx],
                            "dialogue": convo_pair["dialogue_dicts"][model_left_idx][
                                "dialogue"
                            ],
                        },
                        "model_right": {
                            "name": eval_speakers[1 - model_left_idx],
                            "dialogue": convo_pair["dialogue_dicts"][
                                1 - model_left_idx
                            ]["dialogue"],
                        },
                    },
                    "pairing_dict": convo_pair,
                    "pair_id": i,
                }
                if convo_pair.get("is_onboarding"):
                    self.onboarding_tasks.append(task)
                else:
                    self.desired_tasks.append(task)

    def _setup_task_queue(self):
        """
        Fill task queue with conversation pairs.
        """
        for _i in range(self.args.blueprint.annotations_per_pair):
            all_task_keys = list(range(len(self.desired_tasks)))
            random.shuffle(all_task_keys)
            for p_id in all_task_keys:
                self.task_queue.put(self.desired_tasks[p_id])

    def _get_dialogue_ids(self, task: Dict[str, Any]) -> List[int]:
        """
        Return the ids for the dialogues corresponding to a given task.

        :return dialogue_ids:
            A list of two ids which correspond to the id for each conversation
        """
        return task["pairing_dict"]["dialogue_ids"]

    def _poll_task_queue(
        self, worker_id: str, task_data: List[Dict[str, Any]]
    ) -> List[PairingsDict]:
        """
        Poll task queue for tasks for a worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
        worker_data = self._get_worker_data(worker_id)
        num_attempts = 0
        while (not self.task_queue.empty()) and num_attempts < self.task_queue.qsize():
            try:
                next_task = self.task_queue.get()
            except queue.Empty:
                break
            num_attempts += 1

            pair_id = next_task["pair_id"]
            dialogue_ids = self._get_dialogue_ids(next_task)

            # make sure worker has not seen these conversations before
            if pair_id not in worker_data["tasks_completed"] and all(
                d_id not in worker_data["conversations_seen"] for d_id in dialogue_ids
            ):
                # track tasks and conversations seen
                worker_data["tasks_completed"].append(pair_id)
                worker_data["conversations_seen"].extend(dialogue_ids)
                task_data.append(next_task)
                if len(task_data) == self.args.blueprint.subtasks_per_unit:
                    return task_data
            else:
                self.task_queue.put(next_task)

        return task_data

    def _top_up_task_data(
        self, worker_id: str, task_data: List[Dict[str, Any]]
    ) -> List[PairingsDict]:
        """
        Top up worker task data.

        This function is called if ``self.task_queue`` is exhausted but
        task_data for the worker is less than the `tasks_per_unit`.

        Make sure that all added tasks have not been seen by the worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
        worker_data = self._get_worker_data(worker_id)
        tasks_still_needed = self.args.blueprint.subtasks_per_unit - len(task_data)
        tasks_remaining = [
            t_id
            for t_id in range(len(self.desired_tasks))
            if t_id not in worker_data["tasks_completed"]
        ]
        # get any pairings with conversations this worker has not seen to fill this hit
        additional_tasks = [
            t
            for t in tasks_remaining
            if all(
                d_id not in worker_data["conversations_seen"]
                for d_id in self._get_dialogue_ids(self.desired_tasks[t])
            )
        ]
        if tasks_still_needed < len(additional_tasks):
            additional_tasks = random.sample(additional_tasks, tasks_still_needed)
        worker_data["tasks_completed"].extend(additional_tasks)

        for t in additional_tasks:
            worker_data["conversations_seen"].extend(
                self._get_dialogue_ids(self.desired_tasks[t])
            )
            task_data.append(self.desired_tasks[t])

        return task_data

    def get_new_task_data(self, worker_id: str) -> List[PairingsDict]:
        """
        Get next task for worker.

        Returns the next onboarding task if worker hasn't finished them all,
        Otherwise finds a task from the queue they haven't seen

        If they've seen everything in the queue, spin up an
        extra task (one that was in the queue and is now saturated)

        :param worker_id:
            worker id

        :return task_data:
            A list of tasks for the worker to complete
        """
        tasks_per_unit = self.args.blueprint.subtasks_per_unit
        # first add onboarding tasks
        task_data = self.get_onboarding_tasks(worker_id)
        logger.debug(f"Onboarding task data gotten: {len(task_data)}")
        if len(task_data) == tasks_per_unit:
            return task_data

        # poll the task queue for more tasks
        task_data = self._poll_task_queue(worker_id, task_data)
        logger.debug(f"Task queue data gotten: {len(task_data)}")
        if len(task_data) == tasks_per_unit:
            return task_data

        # top up the task_data if we don't hit the desired tasks_per_unit
        task_data = self._top_up_task_data(worker_id, task_data)
        logger.debug(f"Topped off data gotten: {len(task_data)}")
        return task_data

    def requeue_task_data(self, worker_id: str, task_data: List[PairingsDict]):
        """
        Return task to task_queue.

        If the task is an onboarding task, indicate that the worker has
        another onboarding task to do.

        :param worker_id:
            worker id of worker who is returning task

        :param task_data:
            list of unfinished tasks to return to the queue.
        """
        worker_data = self._get_worker_data(worker_id)
        for subtask_data in task_data:
            if subtask_data["task_specs"].get("is_onboarding", False):
                worker_data["onboarding_todo"].append(subtask_data["pair_id"])
            else:
                self.task_queue.put(subtask_data)
                try:
                    worker_data["tasks_completed"].remove(subtask_data["pair_id"])
                    for d_id in self._get_dialogue_ids(subtask_data):
                        worker_data["conversations_seen"].remove(d_id)
                except ValueError:
                    # Task may have shown up in worker's task queue twice
                    # due to some unfortunate race condition
                    logger.exception(
                        f"could not remove task from worker {worker_id} history",
                        exc_info=True,
                    )

    def get_onboarding_tasks(self, worker_id: str) -> List[PairingsDict]:
        """
        Get next onboarding task for given worker.

        :param worker_id:
            worker id

        :return:
            A list of onboarding tasks for the worker
        """
        if len(self.onboarding_tasks) == 0:
            return []

        worker_data = self._get_worker_data(worker_id)
        onboarding_todo = worker_data["onboarding_todo"]
        if not onboarding_todo:
            # worker has completed all required onboarding tasks
            return []
        # get onboarding tasks for workers needing them
        num_tasks_to_return = min(
            len(onboarding_todo), self.args.blueprint.subtasks_per_unit
        )
        onboarding_tasks_chosen = onboarding_todo[:num_tasks_to_return]
        worker_data["onboarding_todo"] = onboarding_todo[num_tasks_to_return:]
        return [self.onboarding_tasks[t_id] for t_id in onboarding_tasks_chosen]

    def check_and_update_worker_approval(self, agent: "Agent"):
        """
        Soft block workers who fail onboarding tasks, keep track of their status.

        :param agent:
            Agent that the worker completed the task with.
        """
        worker = agent.get_worker()
        worker_id = worker.db_id
        save_data = agent.state.get_data()
        all_task_data = save_data["inputs"]
        response_data = save_data["outputs"]["final_data"]
        num_onboarding_tasks = 0
        num_correct = 0

        for i in range(len(all_task_data)):
            is_onboarding = all_task_data[i]["pairing_dict"].get("is_onboarding", False)
            if not is_onboarding:
                # not an onboarding task, no need to check correctness
                continue
            worker_response = response_data[i]["speakerChoice"]
            expected_response = all_task_data[i]["pairing_dict"]["correct_answer"]
            num_onboarding_tasks += 1
            if worker_response == expected_response:
                # count correct answers
                num_correct += 1
        if num_onboarding_tasks == 0:
            # no onboarding tasks found
            if worker_id in self.failed_onboard:
                # worker already failed onboarding, add pairings back to queue
                self.requeue_task_data(worker_id, all_task_data)
            return
        if (
            num_correct / num_onboarding_tasks
        ) >= self.args.blueprint.onboarding_threshold:
            # worker passed onboarding
            return
        # worker failed onboarding, soft block and record
        assert (
            self.block_qualification is not None
        ), "Should not be blocking without a block qualification set"
        worker.grant_qualification(self.block_qualification, 1)
        self.failed_onboard.add(worker_id)

    def get_init_data_for_agent(self, agent: "Agent") -> List[PairingsDict]:
        """
        Return the data for an agent already assigned to a particular unit.
        """
        init_state = agent.state.get_init_state()
        if init_state is not None:
            # reconnecting agent, give what we've got
            return init_state
        else:
            worker = agent.get_worker()
            task_data = self.get_new_task_data(worker.db_id)
            agent.state.set_init_state(task_data)
            self.unit_agent_map[agent.get_unit().db_id] = (worker.db_id, task_data)
            return task_data

    def run_unit(self, unit: "Unit", agent: "Agent") -> None:
        """
        Static runners will get the task data, send it to the user, then wait for the
        agent to act (the data to be completed)
        """
        # Frontend implicitly asks for the initialization data, so we just need
        # to wait for a response
        _ = agent.act(timeout=self.assignment_duration_in_seconds)
        if self.args.blueprint.block_on_onboarding_fail:
            # check whether workers failed onboarding
            self.check_and_update_worker_approval(agent)
        logger.info(f"Acute eval done for {agent}")

    def cleanup_unit(self, unit: "Unit") -> None:
        """
        An incomplete task needs to have the contents of that task requeued into the
        overall task queue.
        """
        logger.info(f"Cleaning up unit {unit.db_id}")
        if unit.db_id not in self.unit_agent_map:
            return logger.warning(
                f"Unit {unit.db_id} already appears to have been cleaned up"
            )
        worker_id, task_data = self.unit_agent_map[unit.db_id]
        del self.unit_agent_map[unit.db_id]
        self.requeue_task_data(worker_id, task_data)
