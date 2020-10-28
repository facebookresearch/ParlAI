#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import shutil
import os
import tempfile
import time

from typing import List

from mephisto.server.blueprints.mock.mock_task_runner import MockTaskRunner
from mephisto.server.architects.mock_architect import MockArchitect
from mephisto.providers.mock.mock_provider import MockProvider
from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.task_launcher import TaskLauncher
from mephisto.data_model.test.utils import get_test_task_run
from mephisto.data_model.assignment import InitializationData
from mephisto.data_model.task import TaskRun
from mephisto.core.supervisor import Supervisor, Job
from mephisto.data_model.blueprint import SharedTaskState


from mephisto.server.architects.mock_architect import MockArchitect, MockArchitectArgs
from mephisto.core.hydra_config import MephistoConfig
from mephisto.providers.mock.mock_provider import MockProviderArgs
from mephisto.data_model.task_config import TaskConfigArgs
from omegaconf import OmegaConf

import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
from mephisto.core.hydra_config import register_script_config
from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.operator import Operator
from mephisto.utils.scripts import augment_config_from_db
from omegaconf import DictConfig, OmegaConf

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import (
    AcuteEvalBlueprint,
    AcuteEvalBlueprintArgs,
)
from parlai.crowdsourcing.tasks.acute_eval.run import (
    ScriptConfig,
    defaults,
    TASK_DIRECTORY,
)

from hydra.experimental import compose, initialize

from mephisto.data_model.packet import Packet, PACKET_TYPE_AGENT_ACTION


# Params
DESIRED_INPUTS = [
    {
        "task_specs": {
            "s1_choice": "I would prefer to talk to <Speaker 1>",
            "s2_choice": "I would prefer to talk to <Speaker 2>",
            "question": "Who would you prefer to talk to for a long conversation?",
            "is_onboarding": True,
            "model_left": {
                "name": "modela",
                "dialogue": [
                    {"id": "modela", "text": "Hello how are you?"},
                    {"id": "human_evaluator", "text": "I'm well, how about yourself?"},
                    {"id": "modela", "text": "Good, just reading a book."},
                    {"id": "human_evaluator", "text": "What book are you reading?"},
                    {
                        "id": "modela",
                        "text": "An English textbook. Do you like to read?",
                    },
                    {
                        "id": "human_evaluator",
                        "text": "Yes, I really enjoy reading, but my favorite thing to do is dog walking.",
                    },
                    {
                        "id": "modela",
                        "text": "Do you have a dog? I don't have any pets",
                    },
                    {
                        "id": "human_evaluator",
                        "text": "Yes, I have a labrador poodle mix.",
                    },
                ],
            },
            "model_right": {
                "name": "modelc",
                "dialogue": [
                    {"id": "modelc", "text": "Hello hello hello"},
                    {"id": "human_evaluator", "text": "How are you?"},
                    {"id": "modelc", "text": "Hello hello hello"},
                    {"id": "human_evaluator", "text": "Hello back"},
                    {"id": "modelc", "text": "Hello hello hello"},
                    {"id": "human_evaluator", "text": "You must really like that word"},
                    {"id": "modelc", "text": "Hello hello hello"},
                    {"id": "human_evaluator", "text": "Ok"},
                ],
            },
        },
        "pairing_dict": {
            "is_onboarding": True,
            "speakers_to_eval": ["modela", "modelc"],
            "correct_answer": "modela",
            "tags": ["onboarding1"],
            "dialogue_dicts": [
                {
                    "speakers": ["modela", "human_evaluator"],
                    "id": "ABCDEF",
                    "evaluator_id_hashed": "HUMAN1",
                    "oz_id_hashed": None,
                    "dialogue": [
                        {"id": "modela", "text": "Hello how are you?"},
                        {
                            "id": "human_evaluator",
                            "text": "I'm well, how about yourself?",
                        },
                        {"id": "modela", "text": "Good, just reading a book."},
                        {"id": "human_evaluator", "text": "What book are you reading?"},
                        {
                            "id": "modela",
                            "text": "An English textbook. Do you like to read?",
                        },
                        {
                            "id": "human_evaluator",
                            "text": "Yes, I really enjoy reading, but my favorite thing to do is dog walking.",
                        },
                        {
                            "id": "modela",
                            "text": "Do you have a dog? I don't have any pets",
                        },
                        {
                            "id": "human_evaluator",
                            "text": "Yes, I have a labrador poodle mix.",
                        },
                    ],
                },
                {
                    "speakers": ["modelc", "human_evaluator"],
                    "id": "ZYX",
                    "evaluator_id_hashed": "HUMAN3",
                    "oz_id_hashed": None,
                    "dialogue": [
                        {"id": "modelc", "text": "Hello hello hello"},
                        {"id": "human_evaluator", "text": "How are you?"},
                        {"id": "modelc", "text": "Hello hello hello"},
                        {"id": "human_evaluator", "text": "Hello back"},
                        {"id": "modelc", "text": "Hello hello hello"},
                        {
                            "id": "human_evaluator",
                            "text": "You must really like that word",
                        },
                        {"id": "modelc", "text": "Hello hello hello"},
                        {"id": "human_evaluator", "text": "Ok"},
                    ],
                },
            ],
        },
        "pair_id": 0,
    },
    {
        "task_specs": {
            "s1_choice": "I would prefer to talk to <Speaker 1>",
            "s2_choice": "I would prefer to talk to <Speaker 2>",
            "question": "Who would you prefer to talk to for a long conversation?",
            "is_onboarding": False,
            "model_left": {
                "name": "modelb",
                "dialogue": [
                    {
                        "id": "human_evaluator",
                        "text": "Hi, I love food, what about you?",
                    },
                    {
                        "id": "modelb",
                        "text": "I love food too, what's your favorite? Mine is burgers.",
                    },
                    {
                        "id": "human_evaluator",
                        "text": "I'm a chef and I love all foods. What do you do?",
                    },
                    {"id": "modelb", "text": "I'm retired now, but I was a nurse."},
                    {
                        "id": "human_evaluator",
                        "text": "Wow, that's really admirable. My sister is a nurse.",
                    },
                    {"id": "modelb", "text": "Do you have any hobbies?"},
                    {"id": "human_evaluator", "text": "I like to paint and play piano"},
                    {
                        "id": "modelb",
                        "text": "You're very artistic. I wish I could be so creative.",
                    },
                ],
            },
            "model_right": {
                "name": "modela",
                "dialogue": [
                    {"id": "modela", "text": "Hi how are you doing?"},
                    {"id": "human_evaluator", "text": "I'm doing ok."},
                    {"id": "modela", "text": "Oh, what's wrong?"},
                    {
                        "id": "human_evaluator",
                        "text": "Feeling a bit sick after my workout",
                    },
                    {"id": "modela", "text": "Do you workout a lot?"},
                    {
                        "id": "human_evaluator",
                        "text": "Yes, I go to the gym every day. I do a lot of lifting.",
                    },
                    {"id": "modela", "text": "That's cool, I like to climb."},
                    {"id": "human_evaluator", "text": "I've never been."},
                ],
            },
        },
        "pairing_dict": {
            "is_onboarding": False,
            "speakers_to_eval": ["modelb", "modela"],
            "tags": ["example1"],
            "dialogue_ids": [0, 1],
            "dialogue_dicts": [
                {
                    "speakers": ["modelb", "human_evaluator"],
                    "id": "AGHIJK",
                    "evaluator_id_hashed": "HUMAN2",
                    "oz_id_hashed": None,
                    "dialogue": [
                        {
                            "id": "human_evaluator",
                            "text": "Hi, I love food, what about you?",
                        },
                        {
                            "id": "modelb",
                            "text": "I love food too, what's your favorite? Mine is burgers.",
                        },
                        {
                            "id": "human_evaluator",
                            "text": "I'm a chef and I love all foods. What do you do?",
                        },
                        {"id": "modelb", "text": "I'm retired now, but I was a nurse."},
                        {
                            "id": "human_evaluator",
                            "text": "Wow, that's really admirable. My sister is a nurse.",
                        },
                        {"id": "modelb", "text": "Do you have any hobbies?"},
                        {
                            "id": "human_evaluator",
                            "text": "I like to paint and play piano",
                        },
                        {
                            "id": "modelb",
                            "text": "You're very artistic. I wish I could be so creative.",
                        },
                    ],
                },
                {
                    "speakers": ["modela", "human_evaluator"],
                    "id": "123456",
                    "evaluator_id_hashed": "HUMAN1",
                    "oz_id_hashed": None,
                    "dialogue": [
                        {"id": "modela", "text": "Hi how are you doing?"},
                        {"id": "human_evaluator", "text": "I'm doing ok."},
                        {"id": "modela", "text": "Oh, what's wrong?"},
                        {
                            "id": "human_evaluator",
                            "text": "Feeling a bit sick after my workout",
                        },
                        {"id": "modela", "text": "Do you workout a lot?"},
                        {
                            "id": "human_evaluator",
                            "text": "Yes, I go to the gym every day. I do a lot of lifting.",
                        },
                        {"id": "modela", "text": "That's cool, I like to climb."},
                        {"id": "human_evaluator", "text": "I've never been."},
                    ],
                },
            ],
        },
        "pair_id": 1,
    },
]
DESIRED_OUTPUTS = {
    "final_data": [
        {"speakerChoice": "modela", "textReason": "Turn 1"},
        {"speakerChoice": "modelb", "textReason": "Turn 2"},
        {"speakerChoice": "modelb", "textReason": "Turn 3"},
        {"speakerChoice": "modelb", "textReason": "Turn 4"},
        {"speakerChoice": "modelb", "textReason": "Turn 5"},
    ]
}


# @dataclass
# class TestScriptConfig(ScriptConfig):
#     defaults: List[Any] = field(default_factory=lambda: test_defaults)


# test_defaults = defaults + [
#     {"mephisto/architect": "mock"},
#     {"mephisto/provider": "mock"},
# ]

relative_task_directory = os.path.relpath(TASK_DIRECTORY, os.path.dirname(__file__))
relative_config_path = os.path.join(relative_task_directory, 'conf')


EMPTY_STATE = SharedTaskState()

# register_script_config(name="test_script_config", module=TestScriptConfig)


# @hydra.main(config_path=relative_task_directory, config_name="test_script_config")
# def main(cfg: DictConfig):

#     # Set up the mock server
#     # data_dir = tempfile.mkdtemp()
#     # database_path = os.path.join(data_dir, "mephisto.db")
#     # db = LocalMephistoDB(database_path)
#     global config
#     config = cfg
#     # cfg = augment_config_from_db(cfg, db)
#     # cfg.mephisto.architect.should_run_server = True
#     print(OmegaConf.to_yaml(config))

#     # import pdb; pdb.set_trace()
#     # return cfg


class TestAcuteEval(unittest.TestCase):
    """
    Test the ACUTE-Eval crowdsourcing task.
    """

    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        database_path = os.path.join(self.data_dir, "mephisto.db")
        self.db = LocalMephistoDB(database_path)
        self.task_id = self.db.new_task(
            "test_acute_eval", AcuteEvalBlueprint.BLUEPRINT_TYPE
        )
        self.task_run_id = get_test_task_run(self.db)
        self.task_run = TaskRun(self.db, self.task_run_id)

        architect_config = OmegaConf.structured(
            MephistoConfig(architect=MockArchitectArgs(should_run_server=True))
        )

        self.architect = MockArchitect(
            self.db, architect_config, EMPTY_STATE, self.task_run, self.data_dir
        )
        self.architect.prepare()
        self.architect.deploy()
        self.urls = self.architect._get_socket_urls()  # FIXME
        self.url = self.urls[0]
        self.provider = MockProvider(self.db)
        self.provider.setup_resources_for_task_run(
            self.task_run, self.task_run.args, EMPTY_STATE, self.url
        )
        self.launcher = TaskLauncher(
            self.db, self.task_run, self.get_mock_assignment_data_array()
        )
        self.launcher.create_assignments()
        self.launcher.launch_units(self.url)
        self.sup = None

        # foo = main()
        # import pdb; pdb.set_trace()

        # Define the configuration settings
        with initialize(config_path=relative_config_path):
            self.config = compose(
                config_name="example",
                overrides=[
                    f'+task_dir={TASK_DIRECTORY}',
                    f'+current_time={int(time.time())}',
                ],
            )
            print(OmegaConf.to_yaml(self.config))  # TODO: remove
        # main()
        # self.config = config
        # self.config = augment_config_from_db(self.config, self.db)
        # self.config.mephisto.architect.should_run_server = True
        # self.config = cfg
        # self.set_config()

    # @hydra.main(config_path=relative_task_directory, config_name="test_script_config")
    # def set_config(cfg: DictConfig):

    def tearDown(self):
        if self.sup is not None:
            self.sup.shutdown()
        self.launcher.expire_units()
        self.architect.cleanup()
        self.architect.shutdown()
        self.db.shutdown()
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def get_mock_assignment_data_array(self) -> List[InitializationData]:
        mock_data = MockTaskRunner.get_mock_assignment_data()
        return [mock_data, mock_data]

    # def test_register_job(self):
    #     """Test registering and running a job run asynchronously"""
    #     # Handle baseline setup
    #     sup = Supervisor(self.db)
    #     self.sup = sup
    #     TaskRunnerClass = AcuteEvalBlueprint.TaskRunnerClass
    #     args = AcuteEvalBlueprint.ArgsClass()
    #     args.timeout_time = 5
    #     # config = OmegaConf.structured(MephistoConfig(blueprint=args))
    #     print(OmegaConf.to_yaml(self.config))
    #     task_runner = TaskRunnerClass(self.task_run, self.config.mephisto, EMPTY_STATE)
    #     sup.register_job(self.architect, task_runner, self.provider)
    #     self.assertEqual(len(sup.channels), 1)
    #     channel_info = list(sup.channels.values())[0]
    #     self.assertIsNotNone(channel_info)
    #     self.assertTrue(channel_info.channel.is_alive())
    #     channel_id = channel_info.channel_id
    #     task_runner = channel_info.job.task_runner
    #     self.assertIsNotNone(channel_id)
    #     self.assertEqual(
    #         len(self.architect.server.subs),
    #         1,
    #         "MockServer doesn't see registered channel",
    #     )
    #     self.assertIsNotNone(
    #         self.architect.server.last_alive_packet,
    #         "No alive packet received by server",
    #     )
    #     sup.launch_sending_thread()
    #     self.assertIsNotNone(sup.sending_thread)

    #     # Register a worker
    #     mock_worker_name = "MOCK_WORKER"
    #     self.architect.server.register_mock_worker(mock_worker_name)
    #     workers = self.db.find_workers(worker_name=mock_worker_name)
    #     self.assertEqual(len(workers), 1, "Worker not successfully registered")
    #     worker = workers[0]

    #     self.architect.server.register_mock_worker(mock_worker_name)
    #     workers = self.db.find_workers(worker_name=mock_worker_name)
    #     self.assertEqual(len(workers), 1, "Worker potentially re-registered")
    #     worker_id = workers[0].db_id

    #     self.assertEqual(len(task_runner.running_assignments), 0)

    #     # Register an agent
    #     mock_agent_details = "FAKE_ASSIGNMENT"
    #     self.architect.server.register_mock_agent(worker_id, mock_agent_details)
    #     agents = self.db.find_agents()
    #     self.assertEqual(len(agents), 1, "Agent was not created properly")

    #     self.architect.server.register_mock_agent(worker_id, mock_agent_details)
    #     agents = self.db.find_agents()
    #     self.assertEqual(len(agents), 1, "Agent may have been duplicated")
    #     agent = agents[0]
    #     self.assertIsNotNone(agent)
    #     self.assertEqual(len(sup.agents), 1, "Agent not registered with supervisor")

    #     self.assertEqual(
    #         len(task_runner.running_assignments), 0, "Task was not yet ready"
    #     )

    #     # # Register another worker
    #     # mock_worker_name = "MOCK_WORKER_2"
    #     # self.architect.server.register_mock_worker(mock_worker_name)
    #     # workers = self.db.find_workers(worker_name=mock_worker_name)
    #     # worker_id = workers[0].db_id

    #     # # Register an agent
    #     # mock_agent_details = "FAKE_ASSIGNMENT_2"
    #     # self.architect.server.register_mock_agent(worker_id, mock_agent_details)

    #     self.assertEqual(
    #         len(task_runner.running_assignments), 1, "Task was not launched"
    #     )
    #     agents = [a.agent for a in sup.agents.values()]

    #     # Make both agents act
    #     agent_id_1, agent_id_2 = agents[0].db_id, agents[1].db_id
    #     agent_1_data = agents[0].datastore.agent_data[agent_id_1]
    #     agent_2_data = agents[1].datastore.agent_data[agent_id_2]
    #     self.architect.server.send_agent_act(agent_id_1, {"text": "message1"})
    #     self.architect.server.send_agent_act(agent_id_2, {"text": "message2"})

    #     # Give up to 1 seconds for the actual operation to occur
    #     start_time = time.time()
    #     TIMEOUT_TIME = 1
    #     while time.time() - start_time < TIMEOUT_TIME:
    #         if len(agent_1_data["acts"]) > 0:
    #             break
    #         time.sleep(0.1)

    #     self.assertLess(
    #         time.time() - start_time, TIMEOUT_TIME, "Did not process messages in time"
    #     )

    #     # Give up to 1 seconds for the task to complete afterwards
    #     start_time = time.time()
    #     TIMEOUT_TIME = 1
    #     while time.time() - start_time < TIMEOUT_TIME:
    #         if len(task_runner.running_assignments) == 0:
    #             break
    #         time.sleep(0.1)
    #     self.assertLess(
    #         time.time() - start_time, TIMEOUT_TIME, "Did not complete task in time"
    #     )

    #     # Give up to 1 seconds for all messages to propogate
    #     start_time = time.time()
    #     TIMEOUT_TIME = 1
    #     while time.time() - start_time < TIMEOUT_TIME:
    #         if len(self.architect.server.received_messages) == 2:
    #             break
    #         time.sleep(0.1)
    #     self.assertLess(
    #         time.time() - start_time, TIMEOUT_TIME, "Not all actions observed in time"
    #     )

    #     sup.shutdown()
    #     self.assertTrue(channel_info.channel.is_closed())

    def test_base_task(self):
        # Handle baseline setup
        sup = Supervisor(self.db)
        self.sup = sup
        TaskRunnerClass = AcuteEvalBlueprint.TaskRunnerClass
        args = AcuteEvalBlueprint.ArgsClass()
        args.timeout_time = 5
        args.is_concurrent = False
        # config = OmegaConf.structured(MephistoConfig(blueprint=args))
        print(OmegaConf.to_yaml(self.config))
        task_runner = TaskRunnerClass(self.task_run, self.config.mephisto, EMPTY_STATE)
        sup.register_job(self.architect, task_runner, self.provider)
        self.assertEqual(len(sup.channels), 1)
        channel_info = list(sup.channels.values())[0]
        self.assertIsNotNone(channel_info)
        self.assertTrue(channel_info.channel.is_alive)
        channel_id = channel_info.channel_id
        task_runner = channel_info.job.task_runner
        self.assertIsNotNone(channel_id)
        self.assertEqual(
            len(self.architect.server.subs),
            1,
            "MockServer doesn't see registered channel",
        )
        self.assertIsNotNone(
            self.architect.server.last_alive_packet,
            "No alive packet received by server",
        )
        sup.launch_sending_thread()
        self.assertIsNotNone(sup.sending_thread)

        # Register a worker
        mock_worker_name = "MOCK_WORKER"
        self.architect.server.register_mock_worker(mock_worker_name)
        workers = self.db.find_workers(worker_name=mock_worker_name)
        self.assertEqual(len(workers), 1, "Worker not successfully registered")
        worker = workers[0]

        self.architect.server.register_mock_worker(mock_worker_name)
        workers = self.db.find_workers(worker_name=mock_worker_name)
        self.assertEqual(len(workers), 1, "Worker potentially re-registered")
        worker_id = workers[0].db_id

        self.assertEqual(len(task_runner.running_assignments), 0)

        # Register an agent
        mock_agent_details = "FAKE_ASSIGNMENT"
        self.architect.server.register_mock_agent(worker_id, mock_agent_details)
        agents = self.db.find_agents()
        agents[0].state = AcuteEvalBlueprint.AgentStateClass(agents[0])
        # By default, the Agent is created with the MockAgentState
        self.assertEqual(len(agents), 1, "Agent was not created properly")

        # self.architect.server.register_mock_agent(worker_id, mock_agent_details)
        # agents = self.db.find_agents()
        # self.assertEqual(len(agents), 1, "Agent may have been duplicated")
        # agent = agents[0]
        # self.assertIsNotNone(agent)
        # self.assertEqual(len(sup.agents), 1, "Agent not registered with supervisor")

        # self.assertEqual(
        #     len(task_runner.running_units), 1, "Ready task was not launched"
        # )

        # # Register another worker
        # mock_worker_name = "MOCK_WORKER_2"
        # self.architect.server.register_mock_worker(mock_worker_name)
        # workers = self.db.find_workers(worker_name=mock_worker_name)
        # worker_id = workers[0].db_id
        # # TODO: only register 1 worker!

        # # Register an agent
        # mock_agent_details = "FAKE_ASSIGNMENT_2"
        # self.architect.server.register_mock_agent(worker_id, mock_agent_details)

        # self.assertEqual(len(task_runner.running_units), 2, "Tasks were not launched")
        # agents = [a.agent for a in sup.agents.values()]

        # Set initial data
        _ = task_runner.get_init_data_for_agent(agents[0])
        # agents[0].state.state['inputs'] = agents[0].state.init_state
        # print(agents[0].state)
        # We need to set this manually because the MockAgentState sets the init state to
        # a different place than where the AcuteEvalRunner expects
        # print('AGENT STATE: ', agents[0].state.state)

        # Make agent act
        agent_id_1 = agents[0].db_id
        # agent_id_1, agent_id_2 = agents[0].db_id, agents[1].db_id
        agent_1_data = agents[0].datastore.agent_data[agent_id_1]
        # agent_2_data = agents[1].datastore.agent_data[agent_id_2]
        # self.architect.server.send_agent_act(agent_id_1, {"text": "message1"})
        # self.architect.server.send_agent_act(agent_id_2, {"text": "message2"})
        packet = Packet(
            packet_type=PACKET_TYPE_AGENT_ACTION,
            sender_id=agent_id_1,
            receiver_id="Mephisto",
            data={"MEPHISTO_is_submit": True, "task_data": DESIRED_OUTPUTS},
        )
        agents[0].observe(packet)
        # import pdb; pdb.set_trace()  # TODO: remove

        # Check that the inputs and outputs are as expected
        state = agents[0].state.get_data()
        self.assertEqual(DESIRED_INPUTS, state['inputs'])
        self.assertEqual(DESIRED_OUTPUTS, state['outputs'])
        print(state)  # TODO: remove

        # # Give up to 1 seconds for the actual operations to occur
        # start_time = time.time()
        # TIMEOUT_TIME = 1
        # while time.time() - start_time < TIMEOUT_TIME:
        #     if len(agent_1_data["acts"]) > 0:
        #         break
        #     time.sleep(0.1)

        # self.assertLess(
        #     time.time() - start_time, TIMEOUT_TIME, "Did not process messages in time"
        # )

        # # Give up to 1 seconds for the task to complete afterwards
        # start_time = time.time()
        # TIMEOUT_TIME = 1
        # while time.time() - start_time < TIMEOUT_TIME:
        #     if len(task_runner.running_units) == 0:
        #         break
        #     time.sleep(0.1)
        # self.assertLess(
        #     time.time() - start_time, TIMEOUT_TIME, "Did not complete task in time"
        # )

        # # Give up to 1 seconds for all messages to propogate
        # start_time = time.time()
        # TIMEOUT_TIME = 1
        # while time.time() - start_time < TIMEOUT_TIME:
        #     if len(self.architect.server.received_messages) == 1:
        #         break
        #     time.sleep(0.1)
        # self.assertLess(
        #     time.time() - start_time, TIMEOUT_TIME, "Not all actions observed in time"
        # )

        sup.shutdown()
        self.assertTrue(channel_info.channel.is_closed)


if __name__ == "__main__":
    unittest.main()
