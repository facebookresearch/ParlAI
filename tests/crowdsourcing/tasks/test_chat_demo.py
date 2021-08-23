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
                "packet_type": "update_status",
                "sender_id": "mephisto",
                "receiver_id": "1",
                "data": {"state": {"agent_display_name": "Chat Agent 1"}},
                "timestamp": 1604343628.610868,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "1",
                "receiver_id": "Mephisto",
                "data": {
                    "text": "Hi! How are you?",
                    "task_data": {},
                    "id": "Chat Agent 1",
                    "episode_done": False,
                    "message_id": "cae52060-800a-4f85-b654-03e60755705a",
                },
                "timestamp": 1604343659.7957256,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "1",
                "data": {
                    "text": "I'm pretty good - you?",
                    "task_data": {},
                    "id": "Chat Agent 2",
                    "episode_done": False,
                    "message_id": "7daabd84-96f8-4a5a-a105-c229ec03c871",
                },
                "timestamp": 1604343667.6145806,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "1",
                "receiver_id": "Mephisto",
                "data": {
                    "text": "I'm okay - how was your weekend?",
                    "task_data": {},
                    "id": "Chat Agent 1",
                    "episode_done": False,
                    "message_id": "b2b4c92d-8b2e-4418-a14a-e1b4dba42a09",
                },
                "timestamp": 1604343676.5881488,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "1",
                "data": {
                    "text": "I was fine. Did you do anything fun?",
                    "task_data": {},
                    "id": "Chat Agent 2",
                    "episode_done": False,
                    "message_id": "389280ae-4a91-466e-8409-7818a4bcf324",
                },
                "timestamp": 1604343688.4189346,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "1",
                "data": {
                    "id": "Coordinator",
                    "text": "Please fill out the form to complete the chat:",
                    "task_data": {
                        "respond_with_form": [
                            {
                                "type": "choices",
                                "question": "How much did you enjoy talking to this user?",
                                "choices": [
                                    "Not at all",
                                    "A little",
                                    "Somewhat",
                                    "A lot",
                                ],
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
                    "message_id": "c2ec35da-cfb7-447b-9767-5b4fcc9231df",
                },
                "timestamp": 1604343688.4194062,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "1",
                "receiver_id": "Mephisto",
                "data": {
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
                    "id": "Chat Agent 1",
                    "episode_done": False,
                    "message_id": "5b799128-5f8b-440e-8947-aee6113690d2",
                },
                "timestamp": 1604343698.391118,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "1",
                "data": {
                    "id": "SUBMIT_WORLD_DATA",
                    "WORLD_DATA": {"example_key": "example_value"},
                    "text": "",
                },
                "timestamp": 1604343706.6179278,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "1",
                "receiver_id": "Mephisto",
                "data": {"task_data": {"final_data": {}}, "MEPHISTO_is_submit": True},
                "timestamp": 1604343708.6682835,
            },
        ]
    },
    "inputs": {},
}
EXPECTED_STATE_AGENT_1 = {
    "outputs": {
        "messages": [
            {
                "packet_type": "update_status",
                "sender_id": "mephisto",
                "receiver_id": "2",
                "data": {"state": {"agent_display_name": "Chat Agent 2"}},
                "timestamp": 1604343628.611253,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "2",
                "data": {
                    "text": "Hi! How are you?",
                    "task_data": {},
                    "id": "Chat Agent 1",
                    "episode_done": False,
                    "message_id": "cae52060-800a-4f85-b654-03e60755705a",
                },
                "timestamp": 1604343659.7962258,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "2",
                "receiver_id": "Mephisto",
                "data": {
                    "text": "I'm pretty good - you?",
                    "task_data": {},
                    "id": "Chat Agent 2",
                    "episode_done": False,
                    "message_id": "7daabd84-96f8-4a5a-a105-c229ec03c871",
                },
                "timestamp": 1604343667.6141868,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "2",
                "data": {
                    "text": "I'm okay - how was your weekend?",
                    "task_data": {},
                    "id": "Chat Agent 1",
                    "episode_done": False,
                    "message_id": "b2b4c92d-8b2e-4418-a14a-e1b4dba42a09",
                },
                "timestamp": 1604343676.5885365,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "2",
                "receiver_id": "Mephisto",
                "data": {
                    "text": "I was fine. Did you do anything fun?",
                    "task_data": {},
                    "id": "Chat Agent 2",
                    "episode_done": False,
                    "message_id": "389280ae-4a91-466e-8409-7818a4bcf324",
                },
                "timestamp": 1604343688.4185243,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "2",
                "data": {
                    "id": "Coordinator",
                    "text": "Please fill out the form to complete the chat:",
                    "task_data": {
                        "respond_with_form": [
                            {
                                "type": "choices",
                                "question": "How much did you enjoy talking to this user?",
                                "choices": [
                                    "Not at all",
                                    "A little",
                                    "Somewhat",
                                    "A lot",
                                ],
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
                    "message_id": "a8494a4a-1868-4dfb-93ad-2c1bb1574993",
                },
                "timestamp": 1604343688.4199135,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "2",
                "receiver_id": "Mephisto",
                "data": {
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
                    "message_id": "13ab6814-cf47-47f7-92c9-d06739918bd7",
                },
                "timestamp": 1604343706.509115,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "mephisto",
                "receiver_id": "2",
                "data": {
                    "id": "SUBMIT_WORLD_DATA",
                    "WORLD_DATA": {"example_key": "example_value"},
                    "text": "",
                },
                "timestamp": 1604343706.6186645,
            },
            {
                "packet_type": "agent_action",
                "sender_id": "2",
                "receiver_id": "Mephisto",
                "data": {"task_data": {"final_data": {}}, "MEPHISTO_is_submit": True},
                "timestamp": 1604343711.3286684,
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
        BLUEPRINT_TYPE,
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
                '+mephisto.blueprint.world_file=${task_dir}/demo_worlds.py',
                '+mephisto.blueprint.task_description_file=${task_dir}/task_description.html',
                '+mephisto.blueprint.num_conversations=1',
                '+mephisto.task.allowed_concurrent=0',
                '+mephisto.task.assignment_duration_in_seconds=600',
                '+mephisto.task.max_num_concurrent_units=0',
                '+mephisto.task.maximum_units_per_worker=0',
                '+num_turns=3',
                '+turn_timeout=300',
            ]
            # TODO: remove all of these params once Hydra 1.1 is released with support
            #  for recursive defaults
            self._set_up_config(
                blueprint_type=BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )

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
