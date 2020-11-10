#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the chat demo crowdsourcing task.
"""

import unittest

# Desired inputs/outputs
DESIRED_STATE_AGENT_0 = {
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
DESIRED_STATE_AGENT_1 = {
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
AGENT_MESSAGES = [
    ("Hi! How are you?", "I'm pretty good - you?"),
    ("I'm okay - how was your weekend?", "I was fine. Did you do anything fun?"),
]
AGENT_0_DISPLAY_ID = 'Chat Agent 1'
AGENT_1_DISPLAY_ID = 'Chat Agent 2'
FORM_PROMPTS = {
    'agent_0': "How much did you enjoy talking to this user?: A lot\nDo you think this user is a bot or a human?: Definitely a human\nEnter any comment here: Yes\n",
    'agent_1': "How much did you enjoy talking to this user?: Not at all\nDo you think this user is a bot or a human?: Definitely a bot\nEnter any comment here: No\n",
}
FORM_RESPONSES = {
    'agent_0': [
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
    'agent_1': [
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
}
# TODO: move this all to a YAML file given the upcoming pytest regressions framework


try:

    # From the Mephisto repo
    from examples.parlai_chat_task_demo.parlai_test_script import TASK_DIRECTORY
    from mephisto.server.blueprints.parlai_chat.parlai_chat_blueprint import (
        SharedParlAITaskState,
        BLUEPRINT_TYPE,
    )

    from parlai.crowdsourcing.utils.tests import CrowdsourcingTestMixin

    class TestChatDemo(CrowdsourcingTestMixin, unittest.TestCase):
        """
        Test the chat demo crowdsourcing task.
        """

        def test_base_task(self):

            # # Setup

            # Set up the config and database
            overrides = [
                '+mephisto.blueprint.world_file=${task_dir}/demo_worlds.py',
                '+mephisto.blueprint.task_description_file=${task_dir}/task_description.html',
                '+mephisto.blueprint.num_conversations=1',
                '+mephisto.task.allowed_concurrent=0',
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

            # Set up the mock human agents
            agent_0_id, agent_1_id = self._register_mock_agents(num_agents=2)

            # # Feed messages to the agents

            # Set initial data
            self.server.request_init_data(agent_0_id)
            self.server.request_init_data(agent_1_id)

            # Have agents talk to each other
            for agent_0_text, agent_1_text in AGENT_MESSAGES:
                self._send_agent_message(
                    agent_id=agent_0_id,
                    agent_display_id=AGENT_0_DISPLAY_ID,
                    text=agent_0_text,
                )
                self._send_agent_message(
                    agent_id=agent_1_id,
                    agent_display_id=AGENT_1_DISPLAY_ID,
                    text=agent_1_text,
                )

            # Have agents fill out the form
            self.server.send_agent_act(
                agent_id=agent_0_id,
                act_content={
                    'text': FORM_PROMPTS['agent_0'],
                    'task_data': {'form_responses': FORM_RESPONSES['agent_0']},
                    'id': AGENT_0_DISPLAY_ID,
                    'episode_done': False,
                },
            )
            self.server.send_agent_act(
                agent_id=agent_1_id,
                act_content={
                    'text': FORM_PROMPTS['agent_1'],
                    'task_data': {'form_responses': FORM_RESPONSES['agent_1']},
                    'id': AGENT_1_DISPLAY_ID,
                    'episode_done': False,
                },
            )

            # Submit the HIT
            self.server.send_agent_act(
                agent_id=agent_0_id,
                act_content={
                    'task_data': {'final_data': {}},
                    'MEPHISTO_is_submit': True,
                },
            )
            self.server.send_agent_act(
                agent_id=agent_1_id,
                act_content={
                    'task_data': {'final_data': {}},
                    'MEPHISTO_is_submit': True,
                },
            )

            # # Check that the inputs and outputs are as expected

            state_0, state_1 = [
                agent.state.get_data() for agent in self.db.find_agents()
            ]
            actual_and_desired_states = [
                (state_0, DESIRED_STATE_AGENT_0),
                (state_1, DESIRED_STATE_AGENT_1),
            ]
            for actual_state, desired_state in actual_and_desired_states:
                assert actual_state['inputs'] == desired_state['inputs']
                assert len(actual_state['outputs']['messages']) == len(
                    desired_state['outputs']['messages']
                )
                for actual_message, desired_message in zip(
                    actual_state['outputs']['messages'],
                    desired_state['outputs']['messages'],
                ):
                    for key, desired_value in desired_message.items():
                        if key == 'timestamp':
                            pass  # The timestamp will obviously be different
                        elif key == 'data':
                            for key_inner, desired_value_inner in desired_message[
                                key
                            ].items():
                                if key_inner == 'message_id':
                                    pass  # The message ID will be different
                                else:
                                    self.assertEqual(
                                        actual_message[key][key_inner],
                                        desired_value_inner,
                                    )
                        else:
                            self.assertEqual(actual_message[key], desired_value)

        def _send_agent_message(self, agent_id: str, agent_display_id: str, text: str):
            """
            Have the agent specified by agent_id send the specified text with the given
            display ID string.
            """
            act_content = {
                "text": text,
                "task_data": {},
                "id": agent_display_id,
                "episode_done": False,
            }
            self.server.send_agent_act(agent_id=agent_id, act_content=act_content)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
