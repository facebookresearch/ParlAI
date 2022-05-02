#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the chat demo crowdsourcing task.
"""

import json
import os
import unittest
from typing import List

import pytest
from pytest_regressions.data_regression import DataRegressionFixture


TASK_CONFIG_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'task_config'
)
TASK_DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_data')
IN_FLIGHT_CONFIG_NAME = "example_in_flight_qa"


try:

    from parlai.crowdsourcing.tasks.turn_annotations_static.run import TASK_DIRECTORY
    from parlai.crowdsourcing.utils.frontend import build_task
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    class TestTurnAnnotationsStatic(AbstractOneTurnCrowdsourcingTest):
        """
        Test the turn annotations crowdsourcing tasks.
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

        def test_no_in_flight_qa(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):
            """
            Test static turn annotations without in-flight QA.
            """

            self.operator = setup_teardown

            overrides = [
                f'mephisto.blueprint.data_jsonl={TASK_CONFIG_FOLDER}/sample_conversations.jsonl'
            ]
            self._test_turn_annotations_static_task(
                task_data_path=os.path.join(TASK_DATA_FOLDER, 'no_in_flight_qa.json'),
                overrides=overrides,
                data_regression=data_regression,
            )

        def test_in_flight_qa(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):
            """
            Test static turn annotations with in-flight QA.
            """

            self.operator = setup_teardown

            overrides = [
                f'mephisto.blueprint.data_jsonl={TASK_CONFIG_FOLDER}/sample_conversations.jsonl'
            ]
            self._test_turn_annotations_static_task(
                task_data_path=os.path.join(TASK_DATA_FOLDER, 'in_flight_qa.json'),
                overrides=overrides,
                data_regression=data_regression,
                config_name=IN_FLIGHT_CONFIG_NAME,
            )

        def test_in_flight_qa_annotation_file(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):
            """
            Test static turn annotations with in-flight QA and with an annotation file.

            The annotation file will list which turns of which conversations should
            receive annotations.
            """

            self.operator = setup_teardown

            overrides = [
                f'+mephisto.blueprint.annotation_indices_jsonl={TASK_DIRECTORY}/task_config/annotation_indices_example.jsonl',
                f'mephisto.blueprint.data_jsonl={TASK_CONFIG_FOLDER}/sample_conversations_annotation_file.jsonl',
                'mephisto.blueprint.subtasks_per_unit=4',
            ]
            self._test_turn_annotations_static_task(
                task_data_path=os.path.join(
                    TASK_DATA_FOLDER, 'in_flight_qa_annotation_file.json'
                ),
                overrides=overrides,
                data_regression=data_regression,
                config_name=IN_FLIGHT_CONFIG_NAME,
            )

        def _test_turn_annotations_static_task(
            self,
            task_data_path: str,
            overrides: List[str],
            data_regression: DataRegressionFixture,
            config_name: str = 'example',
        ):
            """
            Test the static turn annotations task under specific conditions.

            Pass in parameters that will change depending on how we're testing the
            static turn annotations task.
            """

            # # Load the .json of the task data
            with open(task_data_path) as f:
                task_data = json.load(f)

            # # Setup

            build_task(task_directory=TASK_DIRECTORY)

            self._set_up_config(
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
                config_name=config_name,
            )

            # Set up the operator and server
            self._set_up_server()

            self._test_agent_state(task_data=task_data, data_regression=data_regression)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
