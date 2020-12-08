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

import pandas as pd
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.dataframe_regression import DataFrameRegressionFixture
from pytest_regressions.file_regression import FileRegressionFixture


if True:

    from parlai.crowdsourcing.tasks.fast_acute.run import (
        FastAcuteExecutor,
        __file__ as base_task_run_file,
        ACUTE_EVAL_TASK_DIRECTORY,
        BLUEPRINT_TYPE as BASE_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.fast_acute.run_q_function import (
        QLearningFastAcuteExecutor,
        BLUEPRINT_TYPE as Q_FUNCTION_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest

    FAST_ACUTE_TASK_DIRECTORY = os.path.dirname(os.path.abspath(base_task_run_file))

    class TestFastAcute(AbstractOneTurnCrowdsourcingTest):
        """
        Test the Fast ACUTE crowdsourcing task.
        """

        def setup_method(self):

            # Set up common temp directory
            self.root_dir = tempfile.mkdtemp()

            # Params
            self.common_overrides = [
                '+mephisto.blueprint.acute_eval_type=engaging',
                'mephisto.blueprint.block_on_onboarding_fail=False',
                '+mephisto.blueprint.matchups_per_pair=60',
                '+mephisto.blueprint.num_self_chats=5',
                f'+mephisto.blueprint.onboarding_path={FAST_ACUTE_TASK_DIRECTORY}/task_config/onboarding.json',
                f'+mephisto.blueprint.root_dir={self.root_dir}',
                '+mephisto.blueprint.sufficient_matchups_multiplier=2',
                '+mephisto.task.task_name=acute_eval_test',
            ]
            self.models = ['blender_90m_copy1', 'blender_90m_copy2']
            model_string = ','.join(self.models)
            base_task_overrides = [
                f'+mephisto.blueprint.config_path={FAST_ACUTE_TASK_DIRECTORY}/task_config/model_config.json',
                f'+mephisto.blueprint.models=\"{model_string}\"',
                '+mephisto.blueprint.model_pairs=""',
                '+mephisto.blueprint.selfchat_max_turns=6',
                '+mephisto.blueprint.task=blended_skill_talk',
                '+mephisto.blueprint.use_existing_self_chat_files=True',
            ]
            # TODO: clean this up when Hydra has support for recursive defaults

            # Copy over expected self-chat files
            shutil.copytree(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'self_chats'),
                os.path.join(self.root_dir, 'self_chats'),
            )

            self._set_up_config(
                blueprint_type=BASE_BLUEPRINT_TYPE,
                task_directory=ACUTE_EVAL_TASK_DIRECTORY,
                overrides=self.common_overrides + base_task_overrides,
            )
            self.config.mephisto.blueprint.model_pairs = None
            # TODO: hack to manually set mephisto.blueprint.model_pairs to None. Remove
            #  when Hydra releases support for recursive defaults
            self.base_task_runner = FastAcuteExecutor(self.config)

        def test_base_task(
            self,
            data_regression: DataRegressionFixture,
            dataframe_regression: DataFrameRegressionFixture,
            file_regression: FileRegressionFixture,
        ):

            task_data = {
                "final_data": [
                    {"speakerChoice": "model_2", "textReason": "Makes more sense"},
                    {
                        "speakerChoice": "blender_90m_copy1",
                        "textReason": "Makes more sense",
                    },
                    {
                        "speakerChoice": "blender_90m_copy2",
                        "textReason": "Makes more sense",
                    },
                    {
                        "speakerChoice": "blender_90m_copy1",
                        "textReason": "Makes more sense",
                    },
                    {
                        "speakerChoice": "blender_90m_copy2",
                        "textReason": "Makes more sense",
                    },
                ]
            }

            self.base_task_runner.run_selfchat()
            self.base_task_runner.set_up_acute_eval()
            self.config.mephisto.blueprint = self.base_task_runner.fast_acute_args
            self._set_up_server()

            # Check that the agent state is as it should be
            self._test_agent_state(task_data=task_data, data_regression=data_regression)

            # Run analysis and check outputs
            self.base_task_runner.analyze_results(
                args=f'--mephisto-root {self.database_path}'
            )
            self._check_analysis_outputs(
                outputs_folder=self.base_task_runner.results_path,
                save_prefix='base',
                dataframe_regression=dataframe_regression,
                file_regression=file_regression,
            )

        def test_q_function_task(
            self,
            data_regression: DataRegressionFixture,
            dataframe_regression: DataFrameRegressionFixture,
            file_regression: FileRegressionFixture,
        ):

            task_data = {
                "final_data": [
                    {"speakerChoice": "model_2", "textReason": "Makes more sense"},
                    {
                        "speakerChoice": "blender_90m_copy1",
                        "textReason": "Makes more sense",
                    },
                    {
                        "speakerChoice": "blender_90m_copy1",
                        "textReason": "Makes more sense",
                    },
                    {
                        "speakerChoice": "blender_90m_copy2",
                        "textReason": "Makes more sense",
                    },
                    {
                        "speakerChoice": "blender_90m_copy1",
                        "textReason": "Makes more sense",
                    },
                ]
            }

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
                f'+mephisto.blueprint.config_path={config_path}',
                '+mephisto.blueprint.models=""',
                f'+mephisto.blueprint.model_pairs={self.models[0]}:{self.models[1]}',
            ]
            self._set_up_config(
                blueprint_type=Q_FUNCTION_BLUEPRINT_TYPE,
                task_directory=ACUTE_EVAL_TASK_DIRECTORY,
                overrides=overrides,
            )
            self.config.mephisto.blueprint.models = None
            # TODO: hack to manually set mephisto.blueprint.models to None. Remove when
            #  Hydra releases support for recursive defaults
            runner = QLearningFastAcuteExecutor(self.config)
            runner.set_up_acute_eval()
            self.config.mephisto.blueprint = runner.fast_acute_args
            self._set_up_server()

            # Check that the agent state is as it should be
            self._test_agent_state(task_data=task_data, data_regression=data_regression)

            # Run analysis and check outputs
            runner.analyze_results(args=f'--mephisto-root {self.database_path}')
            self._check_analysis_outputs(
                outputs_folder=runner.results_path,
                save_prefix='q_function',
                dataframe_regression=dataframe_regression,
                file_regression=file_regression,
            )

        def _check_analysis_outputs(
            self,
            outputs_folder: str,
            save_prefix: str,
            dataframe_regression: DataFrameRegressionFixture,
            file_regression: FileRegressionFixture,
        ):
            filenames = os.listdir(outputs_folder)
            for filename in filenames:
                parts = filename.split('.')
                save_name = save_prefix + '__' + '.'.join(parts[:-1])
                if parts[-1] == 'csv':
                    df = pd.read_csv(os.path.join(outputs_folder, filename))
                    dataframe_regression.check(data_frame=df, basename=save_name)
                else:
                    with open(os.path.join(outputs_folder, filename)) as f:
                        contents = f.read()
                    file_regression.check(contents=contents, basename=save_name)

        def teardown_method(self):

            super().teardown_method()

            # Tear down temp file
            shutil.rmtree(self.root_dir)


# except ImportError:
#     pass

if __name__ == "__main__":
    unittest.main()
