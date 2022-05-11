#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the ACUTE-Eval crowdsourcing task.
"""


import unittest

import pytest
from pytest_regressions.data_regression import DataRegressionFixture


try:

    from parlai.crowdsourcing.tasks.acute_eval.run import TASK_DIRECTORY
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    class TestAcuteEval(AbstractOneTurnCrowdsourcingTest):
        """
        Test the ACUTE-Eval crowdsourcing task.
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

            task_data = {
                "final_data": [
                    {"speakerChoice": "modela", "textReason": "Turn 1"},
                    {"speakerChoice": "modelb", "textReason": "Turn 2"},
                    {"speakerChoice": "modelb", "textReason": "Turn 3"},
                    {"speakerChoice": "modelb", "textReason": "Turn 4"},
                    {"speakerChoice": "modelb", "textReason": "Turn 5"},
                ]
            }

            # Set up the config, database, operator, and server
            overrides = ['mephisto.blueprint.block_on_onboarding_fail=False']
            self._set_up_config(task_directory=TASK_DIRECTORY, overrides=overrides)
            self._set_up_server()

            # Check that the agent state is as it should be
            self._test_agent_state(task_data=task_data, data_regression=data_regression)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
