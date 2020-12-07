#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the Fast ACUTE crowdsourcing task.
"""

import os
import shutil
import tempfile
import unittest

from pytest_regressions.data_regression import DataRegressionFixture


try:

    from parlai.crowdsourcing.tasks.fast_acute.run import (
        BLUEPRINT_TYPE as BASE_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.fast_acute.run_q_function import (
        BLUEPRINT_TYPE as Q_FUNCTION_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.fast_acute.run import TASK_DIRECTORY
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    class TestFastAcute(AbstractOneTurnCrowdsourcingTest):
        """
        Test the Fast ACUTE crowdsourcing task.
        """

        def setUp(self):

            # Set up common temp directory
            self.root_dir = tempfile.mkdtemp()

            self.common_overrides = [
                'mephisto.blueprint.block_on_onboarding_fail=False',
                'mephisto.blueprint.num_self_chats=5',
                f'mephisto.blueprint.root_dir={self.root_dir}',
            ]

            # Save expected self-chat files
            # {{{TODO}}}

        def test_self_chat_files(self):
            pass
            # {{{TODO: compare to expected using regressions}}}

        def test_base_task(self, data_regression: DataRegressionFixture):

            # Set up the config, database, operator, and server
            overrides = self.common_overrides + [
                f'mephisto.blueprint.config_path={TASK_DIRECTORY}/task_config/model_config.json',
                'mephisto.blueprint.models=\"blender_90m_copy1,blender_90m_copy2\"',
                'mephisto.blueprint.task=blended_skill_talk',
                'mephisto.blueprint.use_existing_self_chat_files=True',
            ]
            self._set_up_config(
                blueprint_type=BASE_BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )
            self._set_up_server()

            # Check that the agent state is as it should be
            self._test_agent_state(data_regression=data_regression)

        def test_q_function_task(self, data_regression: DataRegressionFixture):

            # Save the config file
            config_path = os.path.join(self.root_dir, 'config.json')
            # {{{TODO: save config file, referring to self-chat files}}}

            # Set up the config, database, operator, and server
            overrides = self.common_overrides + [
                f'mephisto.blueprint.config_path={config_path}',
                'mephisto.blueprint.model_pairs=blender_90m_copy1:blender_90m_copy2',
            ]
            self._set_up_config(
                blueprint_type=Q_FUNCTION_BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )
            self._set_up_server()

            # Check that the agent state is as it should be
            self._test_agent_state(data_regression=data_regression)

        def tearDown(self):

            # Tear down temp file
            shutil.rmtree(self.root_dir)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
