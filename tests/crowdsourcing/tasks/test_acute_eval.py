#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the ACUTE-Eval crowdsourcing task.
"""


import unittest

# Desired inputs/outputs
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
# TODO: move this to a YAML file given the upcoming pytest regressions framework


try:

    from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import (
        BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.acute_eval.run import TASK_DIRECTORY
    from parlai.crowdsourcing.utils.tests import CrowdsourcingTestMixin

    class TestAcuteEval(CrowdsourcingTestMixin, unittest.TestCase):
        """
        Test the ACUTE-Eval crowdsourcing task.
        """

        def test_base_task(self):

            # Set up the config, database, operator, and server
            overrides = [f'mephisto.blueprint.block_on_onboarding_fail={False}']
            self._set_up_config(
                blueprint_type=BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )
            self._set_up_server()

            # Set up the mock human agent
            agent_id = self._register_mock_agents(num_agents=1)[0]

            # Set initial data
            self.server.request_init_data(agent_id)

            # Make agent act
            self.server.send_agent_act(
                agent_id, {"MEPHISTO_is_submit": True, "task_data": DESIRED_OUTPUTS}
            )

            # Check that the inputs and outputs are as expected
            state = self.db.find_agents()[0].state.get_data()
            self.assertEqual(DESIRED_INPUTS, state['inputs'])
            self.assertEqual(DESIRED_OUTPUTS, state['outputs'])


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
