#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the QA data collection crowdsourcing task.
"""

import pytest
from pytest_regressions.data_regression import DataRegressionFixture


# Inputs
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

    class TestQADataCollection(AbstractParlAIChatTest):
        """
        Test the QA data collection crowdsourcing task.
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

        def test_base_task(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):

            self.operator = setup_teardown

            # # Setup

            build_task(task_directory=TASK_DIRECTORY)

            # Set up the config and database
            overrides = ['+turn_timeout=300']

            self._set_up_config(task_directory=TASK_DIRECTORY, overrides=overrides)

            # Set up the operator and server
            teacher = get_teacher(self.config)
            world_opt = {"turn_timeout": self.config.turn_timeout, "teacher": teacher}
            shared_state = SharedParlAITaskState(
                world_opt=world_opt, onboarding_world_opt=world_opt
            )
            self._set_up_server(shared_state=shared_state)

            self._test_agent_states(
                num_agents=1,
                agent_display_ids=AGENT_DISPLAY_IDS,
                agent_messages=AGENT_MESSAGES,
                form_messages=FORM_MESSAGES,
                form_task_data=FORM_TASK_DATA,
                data_regression=data_regression,
            )

except ImportError:
    pass
