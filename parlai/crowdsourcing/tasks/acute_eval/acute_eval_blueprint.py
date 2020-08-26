#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.data_model.blueprint import Blueprint
from mephisto.data_model.assignment import InitializationData
from parlai.crowdsourcing.tasks.acute_eval.acute_eval_agent_state import (
    AcuteEvalAgentState,
)
from parlai.crowdsourcing.tasks.acute_eval.acute_eval_runner import AcuteEvalRunner
from parlai.crowdsourcing.tasks.acute_eval.acute_eval_builder import AcuteEvalBuilder
from mephisto.core.registry import register_mephisto_abstraction

import os
import math

from typing import ClassVar, List, Type, Any, Dict, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from mephisto.data_model.blueprint import AgentState, TaskRunner, TaskBuilder
    from argparse import _ArgumentGroup as ArgumentGroup

BLUEPRINT_TYPE = "acute_eval"


# WISH AcuteEval's blueprint can probably be extended to compare more than just convos
@register_mephisto_abstraction()
class AcuteEvalBlueprint(Blueprint):
    """
    Blueprint for a task that asks humans to compare conversational outputs.
    """

    AgentStateClass: ClassVar[Type["AgentState"]] = AcuteEvalAgentState
    TaskBuilderClass: ClassVar[Type["TaskBuilder"]] = AcuteEvalBuilder
    TaskRunnerClass: ClassVar[Type["TaskRunner"]] = AcuteEvalRunner
    supported_architects: ClassVar[List[str]] = ["mock"]  # TODO update
    BLUEPRINT_TYPE = BLUEPRINT_TYPE

    @classmethod
    def assert_task_args(cls, opts: Any) -> None:
        """
        Ensure that the data can be properly loaded.
        """
        if opts.get("pairings_filepath") is not None:
            pairings_filepath = os.path.expanduser(opts["pairings_filepath"])
            assert os.path.exists(
                pairings_filepath
            ), f"Provided file {pairings_filepath} doesn't exist"
        elif opts.get("pairings_task_data") is not None:
            assert (
                len(opts.get("pairings_task_data")) > 0
            ), "Length of data dict provided was 0"
        else:
            raise AssertionError(
                "Must provide one of a data csv, json, or a list of tasks"
            )

        if opts.get("block_on_onboarding_fail") is True:
            if opts.get("block_qualification") is None:
                raise AssertionError(
                    "Must provide `block_qualification` to use `block_on_onboarding_fail`"
                )

    @classmethod
    def add_args_to_group(cls, group: "ArgumentGroup") -> None:
        """
        Adds required options for AcuteEvalBlueprints.

        task_source points to the file intending to be deployed for this task
        pairings_filepath has the data to be deployed for this task.
        """
        super(AcuteEvalBlueprint, cls).add_args_to_group(group)

        group.description = """
            AcuteEvalBlueprint: Tasks launched from acute eval blueprints
            require sets of pairings for workers to be able to compare to.

            These pairings can be provided as a csv or by passing a
            pairings_task_data dict into extra_args.
        """
        group.add_argument(
            "--annotations-per-pair",
            dest="annotations_per_pair",
            type=int,
            default=1,
            help="Number of annotations per conversation comparison pair",
        )
        group.add_argument(
            "--pairings-filepath",
            dest="pairings_filepath",
            type=str,
            default=None,
            help="path to the file containing the task dictionaries",
        )
        group.add_argument(
            "--s1-choice",
            dest="s1_choice",
            type=str,
            default="I would prefer to talk to <Speaker 1>",
            help="text next to speaker 1 radio button",
        )
        group.add_argument(
            "--s2-choice",
            dest="s2_choice",
            type=str,
            default="I would prefer to talk to <Speaker 2>",
            help="text next to speaker 2 radio button",
        )
        group.add_argument(
            "--eval-question",
            dest="eval_question",
            type=str,
            default="Who would you prefer to talk to for a long conversation?",
            help='question to present to turker for comparison (e.g. "Which speaker is better?")',
        )
        group.add_argument(
            "--block-on-onboarding-fail",
            dest="block_on_onboarding_fail",
            type=bool,
            default=True,
            help="whether to block on onboarding failure",
        )
        group.add_argument(
            "--subtasks-per-unit",
            dest="subtasks_per_unit",
            type=int,
            default=5,
            help="number of subtasks/comparisons to do per unit",
        )
        group.add_argument(
            "--onboarding-threshold",
            dest="onboarding_threshold",
            type=float,
            default=0.75,
            help="minimum accuracy on onboarding tasks, as a float 0-1.0",
        )
        group.add_argument(
            "--random-seed",
            dest="random_seed",
            type=int,
            default=42,
            help="seed for random",
        )
        group.add_argument(
            "--additional-task-description",
            dest="additional_task_description",
            type=str,
            default='',
            help="Additional text to show on the left pane",
        )
        return

    def get_frontend_args(self) -> Dict[str, Any]:
        """
        Specifies what options within a task_config should be forwarded to the client
        for use by the task's frontend.
        """
        return {
            "task_description": "Placeholder Task Description - Javascript failed to load",
            "frame_height": 650,
            "num_subtasks": self.opts["subtasks_per_unit"],
            "question": self.opts["eval_question"],
            "block_mobile": True,
            "get_task_feedback": False,  # TODO(#95) make option
            "additional_task_description": self.opts['additional_task_description'],
        }

    def get_initialization_data(self) -> Iterable["InitializationData"]:
        """
        Return the InitializationData retrieved from the specified stream.
        """
        # TODO(#99) once we can release HITs over time, configure this to
        # release as many as needed thusfar and top off when
        # onboardings fail
        print(self.opts)
        num_conversations = math.ceil(
            self.opts.get("num_matchup_pairs", 8)
            / max((self.opts["subtasks_per_unit"] - 1), 1)
        )  # release enough hits to finish all annotations requested
        return [
            InitializationData(shared={}, unit_data=[{}])
            for d in range(num_conversations)
        ]
