#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the chat demo crowdsourcing task.
"""

import os
import unittest

import parlai.utils.testing as testing_utils


# Desired inputs/outputs
EXPECTED_STATE_AGENT_0 = {
    "outputs": {
        "messages": [
            {
                "text": "Hi! How are you?",
                "task_data": {},
                "id": "Chat Agent 1",
                "episode_done": False,
            },
            {
                "text": "I'm pretty good - you?",
                "task_data": {},
                "id": "Chat Agent 2",
                "episode_done": False,
            },
            {
                "text": "I'm okay - how was your weekend?",
                "task_data": {},
                "id": "Chat Agent 1",
                "episode_done": False,
            },
            {
                "text": "I was fine. Did you do anything fun?",
                "task_data": {},
                "id": "Chat Agent 2",
                "episode_done": False,
            },
            {
                "id": "Coordinator",
                "text": "Please fill out the form to complete the chat:",
                "task_data": {
                    "respond_with_form": [
                        {
                            "type": "choices",
                            "question": "How much did you enjoy talking to this user?",
                            "choices": ["Not at all", "A little", "Somewhat", "A lot"],
                        },
                        {
                            "type": "choices",
                            "question": "Do you think this user is a bot or a human?",
                            "choices": [
                                "Definitely a bot",
                                "Probably a bot",
                                "Probably a human",
                                "Definitely a human",
                            ],
                        },
                        {"type": "text", "question": "Enter any comment here"},
                    ]
                },
            },
            {
                "text": "How much did you enjoy talking to this user?: A lot\nDo you think this user is a bot or a human?: Definitely a human\nEnter any comment here: Yes\n",
                "task_data": {
                    "form_responses": [
                        {
                            "question": "How much did you enjoy talking to this user?",
                            "response": "A lot",
                        },
                        {
                            "question": "Do you think this user is a bot or a human?",
                            "response": "Definitely a human",
                        },
                        {"question": "Enter any comment here", "response": "Yes"},
                    ]
                },
            },
            {
                "id": "SUBMIT_WORLD_DATA",
                "WORLD_DATA": {"example_key": "example_value"},
                "text": "",
            },
        ]
    },
    "inputs": {},
}
EXPECTED_STATE_AGENT_1 = {
    "outputs": {
        "messages": [
            {
                "text": "Hi! How are you?",
                "task_data": {},
                "id": "Chat Agent 1",
                "episode_done": False,
            },
            {
                "text": "I'm pretty good - you?",
                "task_data": {},
                "id": "Chat Agent 2",
                "episode_done": False,
            },
            {
                "text": "I'm okay - how was your weekend?",
                "task_data": {},
                "id": "Chat Agent 1",
                "episode_done": False,
            },
            {
                "text": "I was fine. Did you do anything fun?",
                "task_data": {},
                "id": "Chat Agent 2",
                "episode_done": False,
            },
            {
                "id": "Coordinator",
                "text": "Please fill out the form to complete the chat:",
                "task_data": {
                    "respond_with_form": [
                        {
                            "type": "choices",
                            "question": "How much did you enjoy talking to this user?",
                            "choices": ["Not at all", "A little", "Somewhat", "A lot"],
                        },
                        {
                            "type": "choices",
                            "question": "Do you think this user is a bot or a human?",
                            "choices": [
                                "Definitely a bot",
                                "Probably a bot",
                                "Probably a human",
                                "Definitely a human",
                            ],
                        },
                        {"type": "text", "question": "Enter any comment here"},
                    ]
                },
            },
            {
                "text": "How much did you enjoy talking to this user?: Not at all\nDo you think this user is a bot or a human?: Definitely a bot\nEnter any comment here: No\n",
                "task_data": {
                    "form_responses": [
                        {
                            "question": "How much did you enjoy talking to this user?",
                            "response": "Not at all",
                        },
                        {
                            "question": "Do you think this user is a bot or a human?",
                            "response": "Definitely a bot",
                        },
                        {"question": "Enter any comment here", "response": "No"},
                    ]
                },
                "id": "Chat Agent 2",
                "episode_done": False,
            },
            {
                "id": "SUBMIT_WORLD_DATA",
                "WORLD_DATA": {"example_key": "example_value"},
                "text": "",
            },
        ]
    },
    "inputs": {},
}
EXPECTED_STATES = (EXPECTED_STATE_AGENT_0, EXPECTED_STATE_AGENT_1)
AGENT_MESSAGES = [
    ("Hi! How are you?", "I'm pretty good - you?"),
    ("I'm okay - how was your weekend?", "I was fine. Did you do anything fun?"),
]
AGENT_DISPLAY_IDS = ('Chat Agent 1', 'Chat Agent 2')
FORM_MESSAGES = (
    "How much did you enjoy talking to this user?: A lot\nDo you think this user is a bot or a human?: Definitely a human\nEnter any comment here: Yes\n",
    "How much did you enjoy talking to this user?: Not at all\nDo you think this user is a bot or a human?: Definitely a bot\nEnter any comment here: No\n",
)
FORM_RESPONSES = (
    [
        {
            "question": "How much did you enjoy talking to this user?",
            "response": "A lot",
        },
        {
            "question": "Do you think this user is a bot or a human?",
            "response": "Definitely a human",
        },
        {"question": "Enter any comment here", "response": "Yes"},
    ],
    [
        {
            "question": "How much did you enjoy talking to this user?",
            "response": "Not at all",
        },
        {
            "question": "Do you think this user is a bot or a human?",
            "response": "Definitely a bot",
        },
        {"question": "Enter any comment here", "response": "No"},
    ],
)
FORM_TASK_DATA = [{'form_responses': responses} for responses in FORM_RESPONSES]
# TODO: move this all to a YAML file given the upcoming pytest regressions framework


try:

    import mephisto
    from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
        SharedParlAITaskState,
    )

    from parlai.crowdsourcing.utils.tests import AbstractParlAIChatTest

    TASK_DIRECTORY = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(mephisto.__file__))),
        'examples',
        'parlai_chat_task_demo',
    )

    class TestChatDemo(AbstractParlAIChatTest, unittest.TestCase):
        """
        Test the chat demo crowdsourcing task.
        """

        # TODO: remove the inheritance from unittest.TestCase once this test uses pytest
        #  regressions. Also use a pytest.fixture to call self._setup() and
        #  self._teardown(), like the other tests use, instead of calling them with
        #  self.setUp() and self.tearDown()

        def setUp(self) -> None:
            self._setup()

        def tearDown(self) -> None:
            self._teardown()

        @testing_utils.retry(ntries=3)
        def test_base_task(self):

            # # Setup

            # Set up the config and database
            overrides = [
                '++mephisto.task.allowed_concurrent=0',
                '++mephisto.task.assignment_duration_in_seconds=600',
                '++mephisto.task.max_num_concurrent_units=0',
                '++mephisto.task.maximum_units_per_worker=0',
                '++num_turns=3',
                '++turn_timeout=300',
            ]
            self._set_up_config(task_directory=TASK_DIRECTORY, overrides=overrides)

            # Set up the operator and server
            world_opt = {
                "num_turns": self.config.num_turns,
                "turn_timeout": self.config.turn_timeout,
            }
            shared_state = SharedParlAITaskState(
                world_opt=world_opt, onboarding_world_opt=world_opt
            )
            self._set_up_server(shared_state=shared_state)

            # Check that the agent states are as they should be
            self._test_agent_states(
                num_agents=2,
                agent_display_ids=AGENT_DISPLAY_IDS,
                agent_messages=AGENT_MESSAGES,
                form_messages=FORM_MESSAGES,
                form_task_data=FORM_TASK_DATA,
                expected_states=EXPECTED_STATES,
            )

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
