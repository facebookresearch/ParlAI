#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the Fast ACUTE crowdsourcing task.
"""

import json
import os
import shutil
import tempfile
import unittest

from pytest_regressions.data_regression import (
    DataRegressionFixture,
    FileRegressionFixture,
)


try:

    from parlai.crowdsourcing.tasks.fast_acute.run import (
        BLUEPRINT_TYPE as BASE_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.fast_acute.run_q_function import (
        BLUEPRINT_TYPE as Q_FUNCTION_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.fast_acute import run
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    TASK_DIRECTORY = os.path.dirname(os.path.abspath(run.__file__))

    class TestFastAcute(AbstractOneTurnCrowdsourcingTest):
        """
        Test the Fast ACUTE crowdsourcing task.
        """

        def setup_class(self):

            super().setup_class()

            # Set up common temp directory
            self.root_dir = tempfile.mkdtemp()

            # Params
            self.common_overrides = [
                '+mephisto.blueprint.block_on_onboarding_fail=False',
                'mephisto.blueprint.num_self_chats=1',   # TODO: change back to 5!!
                f'mephisto.blueprint.root_dir={self.root_dir}',
            ]
            self.models = ['blender_90m_copy1', 'blender_90m_copy2']
            model_string = ','.join(self.models)
            self.base_task_overrides = [
                f'mephisto.blueprint.config_path={TASK_DIRECTORY}/task_config/model_config.json',
                f'+mephisto.blueprint.models=\"{model_string}\"',
                '+mephisto.blueprint.model_pairs=""',
                '+mephisto.blueprint.selfchat_max_turns=6',
                '+mephisto.blueprint.task=blended_skill_talk',
                '+mephisto.blueprint.use_existing_self_chat_files=True',
                '+mephisto.task.task_name=acute_eval_test',
            ]

            # Save expected self-chat files
            self._set_up_config(
                blueprint_type=BASE_BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=self.common_overrides + self.base_task_overrides,
            )
            self.config.mephisto.blueprint.model_pairs = None
            # TODO: hack to manually set mephisto.blueprint.model_pairs to None. Remove
            #  when Hydra releases support for recursive defaults
            self.base_task_runner = run.FastAcuteExecutor(self.config)
            self.base_task_runner.run_selfchat()

        def test_self_chat_files(self, file_regression: FileRegressionFixture):
            for model in self.models:
                outfile = self.base_task_runner._get_selfchat_log_path(model)
                file_regression.check(outfile)

        def test_base_task(self, data_regression: DataRegressionFixture):

            # Set up the config, database, operator, and server
            self._set_up_config(
                blueprint_type=BASE_BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=self.common_overrides + self.base_task_overrides,
            )
            self._set_up_server()

            # Check that the agent state is as it should be
            self._test_agent_state(data_regression=data_regression)

        def test_q_function_task(self, data_regression: DataRegressionFixture):

            # Save the config file
            config_path = os.path.join(self.root_dir, 'config.json')
            config = {}
            for model in self.models:
                config[model] = {
                    'log_path': self.base_task_runner._get_selfchat_log_path(model),
                    'is_selfchat': True,
                }
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Set up the config, database, operator, and server
            assert len(self.models) == 2
            overrides = self.common_overrides + [
                f'mephisto.blueprint.config_path={config_path}',
                f'mephisto.blueprint.model_pairs={self.models[0]}:{self.models[1]}',
            ]
            self._set_up_config(
                blueprint_type=Q_FUNCTION_BLUEPRINT_TYPE,
                task_directory=TASK_DIRECTORY,
                overrides=overrides,
            )
            self._set_up_server()

            # Check that the agent state is as it should be
            self._test_agent_state(data_regression=data_regression)

        def teardown_class(self):

            super().teardown_class()

            # Tear down temp file
            shutil.rmtree(self.root_dir)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
