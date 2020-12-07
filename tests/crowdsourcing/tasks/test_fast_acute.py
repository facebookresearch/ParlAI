#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the Fast ACUTE crowdsourcing task.
"""


import unittest


try:

    from parlai.crowdsourcing.tasks.fast_acute.run import (
        BLUEPRINT_TYPE as BASE_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.fast_acute.run_q_function import (
        BLUEPRINT_TYPE as Q_FUNCTION_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.acute_eval.run import TASK_DIRECTORY
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    class TestFastAcute(AbstractOneTurnCrowdsourcingTest):
        """
        Test the Fast ACUTE crowdsourcing task.
        """

        def test_base_task(self):

            # Set up the config, database, operator, and server
            # TODO: revise below
            overrides = [f'mephisto.blueprint.block_on_onboarding_fail={False}']
            self._set_up_config(
                blueprint_type=BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )
            self._set_up_server()

            # Check that the agent state is as it should be
            expected_state = {'inputs': DESIRED_INPUTS, 'outputs': DESIRED_OUTPUTS}
            self._test_agent_state(expected_state=expected_state)

        def test_q_function_task(self):

            # Set up the config, database, operator, and server
            overrides = [f'mephisto.blueprint.block_on_onboarding_fail={False}']
            self._set_up_config(
                blueprint_type=BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )
            self._set_up_server()

            # Check that the agent state is as it should be
            expected_state = {'inputs': DESIRED_INPUTS, 'outputs': DESIRED_OUTPUTS}
            self._test_agent_state(expected_state=expected_state)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
