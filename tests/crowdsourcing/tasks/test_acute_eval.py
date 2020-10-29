#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import tempfile
import time
import unittest
from typing import List

from hydra.experimental import compose, initialize
from mephisto.core.hydra_config import MephistoConfig
from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.supervisor import Supervisor
from mephisto.core.task_launcher import TaskLauncher
from mephisto.data_model.assignment import InitializationData
from mephisto.data_model.blueprint import SharedTaskState
from mephisto.data_model.packet import Packet, PACKET_TYPE_AGENT_ACTION
from mephisto.data_model.task import TaskRun
from mephisto.data_model.test.utils import get_test_task_run
from mephisto.providers.mock.mock_provider import MockProvider
from mephisto.server.architects.mock_architect import MockArchitect, MockArchitectArgs
from mephisto.server.blueprints.mock.mock_task_runner import MockTaskRunner
from omegaconf import OmegaConf

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import (
    AcuteEvalBlueprint,
)
from parlai.crowdsourcing.tasks.acute_eval.run import TASK_DIRECTORY


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
EMPTY_STATE = SharedTaskState()


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

        # Define the configuration settings
        relative_task_directory = os.path.relpath(
            TASK_DIRECTORY, os.path.dirname(__file__)
        )
        relative_config_path = os.path.join(relative_task_directory, 'conf')
        with initialize(config_path=relative_config_path):
            self.config = compose(
                config_name="example",
                overrides=[
                    f'+task_dir={TASK_DIRECTORY}',
                    f'+current_time={int(time.time())}',
                ],
            )

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

    def test_base_task(self):
        # Handle baseline setup
        sup = Supervisor(self.db)
        self.sup = sup
        task_runner_class = AcuteEvalBlueprint.TaskRunnerClass
        args = AcuteEvalBlueprint.ArgsClass()
        args.timeout_time = 5
        args.is_concurrent = False
        print(OmegaConf.to_yaml(self.config))
        task_runner = task_runner_class(
            self.task_run, self.config.mephisto, EMPTY_STATE
        )
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

        # Set initial data
        _ = task_runner.get_init_data_for_agent(agents[0])

        # Make agent act
        agent_id_1 = agents[0].db_id
        packet = Packet(
            packet_type=PACKET_TYPE_AGENT_ACTION,
            sender_id=agent_id_1,
            receiver_id="Mephisto",
            data={"MEPHISTO_is_submit": True, "task_data": DESIRED_OUTPUTS},
        )
        agents[0].observe(packet)

        # Check that the inputs and outputs are as expected
        state = agents[0].state.get_data()
        self.assertEqual(DESIRED_INPUTS, state['inputs'])
        self.assertEqual(DESIRED_OUTPUTS, state['outputs'])

        sup.shutdown()
        self.assertTrue(channel_info.channel.is_closed)


if __name__ == "__main__":
    unittest.main()
