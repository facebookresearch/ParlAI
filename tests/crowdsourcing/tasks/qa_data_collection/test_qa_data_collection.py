#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the QA data collection crowdsourcing task.
"""

import json
import os
import unittest


# Inputs
# The conversation is not exactly making sense if put in context.
# It is just to have the interaction between the agents simulated.
AGENT_DISPLAY_IDS = ('QA Agent',)
AGENT_MESSAGES = [("Who was the first reigning pope to ever visit the Americas?",)]
FORM_MESSAGES = ("Pope Paul VI",)
FORM_TASK_DATA = ({},)
# No info is sent through the 'task_data' field when submitting the form


try:

    from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
        SharedParlAITaskState,
    )

    from parlai.crowdsourcing.tasks.qa_data_collection.run import TASK_DIRECTORY
    from parlai.crowdsourcing.tasks.qa_data_collection.util import get_teacher
    from parlai.crowdsourcing.utils.frontend import build_task
    from parlai.crowdsourcing.utils.tests import AbstractParlAIChatTest

    class TestQADataCollection(AbstractParlAIChatTest, unittest.TestCase):
        """
        Test the QA data collection crowdsourcing task.
        """

        # TODO: remove the inheritance from unittest.TestCase once this test uses pytest
        #  regressions. Also use a pytest.fixture to call self._setup() and
        #  self._teardown(), like the other tests use, instead of calling them with
        #  self.setUp() and self.tearDown()

        def setUp(self) -> None:
            self._setup()

        def tearDown(self) -> None:
            self._teardown()

        def test_base_task(self):

            # Paths
            expected_states_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'expected_states'
            )
            expected_state_path = os.path.join(expected_states_folder, 'state.json')

            # # Setup

            build_task(task_directory=TASK_DIRECTORY)

            # Set up the config and database
            overrides = ['+turn_timeout=300']

            self._set_up_config(task_directory=TASK_DIRECTORY, overrides=overrides)

            # 'train:ordered' in order to avoid randomness that might break the test.
            self.config.teacher.datatype = 'train:ordered'
            # Set up the operator and server
            teacher = get_teacher(self.config)
            world_opt = {"turn_timeout": self.config.turn_timeout, "teacher": teacher}
            shared_state = SharedParlAITaskState(
                world_opt=world_opt, onboarding_world_opt=world_opt
            )
            self._set_up_server(shared_state=shared_state)

            # Check that the agent states are as they should be
            with open(expected_state_path) as f:
                expected_state = json.load(f)
            self._test_agent_states(
                num_agents=1,
                agent_display_ids=AGENT_DISPLAY_IDS,
                agent_messages=AGENT_MESSAGES,
                form_messages=FORM_MESSAGES,
                form_task_data=FORM_TASK_DATA,
                expected_states=(expected_state,),
            )

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
