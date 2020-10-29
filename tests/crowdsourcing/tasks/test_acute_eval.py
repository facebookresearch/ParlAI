#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import unittest
from typing import ClassVar, Type

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import (
    AcuteEvalBlueprint,
)
from parlai.crowdsourcing.tasks.acute_eval.run import TASK_DIRECTORY

try:
    from mephisto.core.supervisor import Supervisor
    from mephisto.data_model.blueprint import Blueprint
    from mephisto.data_model.packet import Packet, PACKET_TYPE_AGENT_ACTION
    from mephisto.data_model.test.utils import AbstractTestSupervisor, EMPTY_STATE

    SKIP_TESTS = False
except ModuleNotFoundError:
    SKIP_TESTS = True


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


@unittest.skipIf(SKIP_TESTS, "Mephisto not installed.")
class TestAcuteEval(AbstractTestSupervisor, unittest.TestCase):
    """
    Test the ACUTE-Eval crowdsourcing task.
    """

    BlueprintClass: ClassVar[Type["Blueprint"]] = AcuteEvalBlueprint

    def setUp(self):
        super().setUp()

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
        print(OmegaConf.to_yaml(self.config))

    def test_base_task(self):

        # Handle baseline setup
        sup = Supervisor(self.db)
        self.sup = sup
        task_runner_class = self.BlueprintClass.TaskRunnerClass
        args = self.BlueprintClass.ArgsClass()
        args.timeout_time = 5
        args.is_concurrent = False
        task_runner = task_runner_class(
            self.task_run, self.config.mephisto, EMPTY_STATE
        )
        sup.register_job(self.architect, task_runner, self.provider)
        channel_info = list(sup.channels.values())[0]
        task_runner = channel_info.job.task_runner
        sup.launch_sending_thread()

        # Register a worker
        mock_worker_name = "MOCK_WORKER"
        self.architect.server.register_mock_worker(mock_worker_name)
        workers = self.db.find_workers(worker_name=mock_worker_name)
        worker_id = workers[0].db_id

        # Register an agent
        mock_agent_details = "FAKE_ASSIGNMENT"
        self.architect.server.register_mock_agent(worker_id, mock_agent_details)
        agent = self.db.find_agents()[0]
        agent.state = self.BlueprintClass.AgentStateClass(agent)
        # By default, the Agent is created with the MockAgentState

        # Set initial data
        _ = task_runner.get_init_data_for_agent(agent)

        # Make agent act
        agent_id_1 = agent.db_id
        packet = Packet(
            packet_type=PACKET_TYPE_AGENT_ACTION,
            sender_id=agent_id_1,
            receiver_id="Mephisto",
            data={"MEPHISTO_is_submit": True, "task_data": DESIRED_OUTPUTS},
        )
        agent.observe(packet)

        # Check that the inputs and outputs are as expected
        state = agent.state.get_data()
        self.assertEqual(DESIRED_INPUTS, state['inputs'])
        self.assertEqual(DESIRED_OUTPUTS, state['outputs'])

        sup.shutdown()


if __name__ == "__main__":
    unittest.main()
