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


SAMPLE_CONVERSATIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'task_config',
    'sample_conversations.jsonl',
)
EXPECTED_STATES_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'expected_states'
)


try:

    from parlai.crowdsourcing.tasks.turn_annotations_static.run import TASK_DIRECTORY
    from parlai.crowdsourcing.tasks.turn_annotations_static.run_in_flight_qa import (
        TASK_DIRECTORY as TASK_DIRECTORY_IN_FLIGHT_QA,
    )
    from parlai.crowdsourcing.tasks.turn_annotations_static.turn_annotations_blueprint import (
        STATIC_BLUEPRINT_TYPE,
        STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    class TestTurnAnnotationsStatic(AbstractOneTurnCrowdsourcingTest):
        """
        Test the turn annotations crowdsourcing tasks.
        """

        def test_no_in_flight_qa(self):

            # # Load the .json of the expected state
            expected_state_path = os.path.join(
                EXPECTED_STATES_FOLDER, 'no_in_flight_qa.json'
            )
            with open(expected_state_path) as f:
                expected_state = json.load(f)

            # # Setup

            # Set up the config and database
            overrides = [
                '+mephisto.blueprint.annotation_indices_jsonl=null',
                '+mephisto.blueprint.conversation_count=null',
                f'mephisto.blueprint.data_jsonl={SAMPLE_CONVERSATIONS_PATH}',
                'mephisto.blueprint.onboarding_qualification=null',
                '+mephisto.blueprint.random_seed=42',
            ]
            # TODO: remove all of these params once Hydra 1.1 is released with support
            #  for recursive defaults
            # TODO: test onboarding as well, and don't nullify the onboarding_qualification param
            self._set_up_config(
                blueprint_type=STATIC_BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )

            # Set up the operator and server
            self._set_up_server()

            self._test_agent_state(expected_state=expected_state)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
