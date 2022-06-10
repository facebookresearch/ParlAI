#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the chat demo crowdsourcing task.
"""

import os

import parlai.utils.testing as testing_utils
import pytest
from pytest_regressions.data_regression import DataRegressionFixture


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

    class TestChatDemo(AbstractParlAIChatTest):
        """
        Test the chat demo crowdsourcing task.
        """

        @pytest.fixture(scope="function")
        def setup_teardown(self):
            """
            Call code to set up and tear down tests.
            """
            self._setup()
            yield self.operator
            # All code after this will be run upon teardown
            self._teardown()

        @testing_utils.retry(ntries=3)
        def test_base_task(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):

            # # Setup
            self.operator = setup_teardown

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
                data_regression=data_regression,
            )

except ImportError:
    pass
