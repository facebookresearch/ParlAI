#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for DialCrowd.
"""

import json
import os
import unittest

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

TASK_CONFIG_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'task_config'
)
TASK_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_data')

try:
    from parlai.crowdsourcing.tasks.dialcrowd.run import TASK_DIRECTORY
    from parlai.crowdsourcing.utils.frontend import build_task
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    class TestDialCrowd(AbstractOneTurnCrowdsourcingTest):
        """
        Test the DialCrowd tasks.
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

        def test_dialcrowd(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):
            """
            Test DialCrowd.
            """

            self.operator = setup_teardown

            overrides = [
                f'mephisto.blueprint.data_jsonl={TASK_CONFIG_FOLDER}/sample_data.jsonl'
            ]

            task_data_path = os.path.join(TASK_DATA_FOLDER, 'data.json')

            # # Load the .json of the task data
            with open(task_data_path) as f:
                task_data = json.load(f)

            # # Setup

            build_task(task_directory=TASK_DIRECTORY)

            self._set_up_config(
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
                config_name='example',
            )

            # Set up the operator and server
            self._set_up_server()

            self._test_agent_state(task_data=task_data, data_regression=data_regression)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
