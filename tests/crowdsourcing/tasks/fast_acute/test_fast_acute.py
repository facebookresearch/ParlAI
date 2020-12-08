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
import pytest
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

        TASK_DATA = {
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

        @pytest.fixture(scope="class")
        def setup_teardown(self):
            """
            Call code to set up and tear down tests.

            Run this only once because we'll be running all Fast ACUTE code before
            checking any results.
            """

            self._setup()

            # Set up common temp directory
            root_dir = tempfile.mkdtemp()

            # Params
            common_overrides = [
                '+mephisto.blueprint.acute_eval_type=engaging',
                'mephisto.blueprint.block_on_onboarding_fail=False',
                '+mephisto.blueprint.matchups_per_pair=60',
                '+mephisto.blueprint.num_self_chats=5',
                f'+mephisto.blueprint.onboarding_path={FAST_ACUTE_TASK_DIRECTORY}/task_config/onboarding.json',
                f'+mephisto.blueprint.root_dir={root_dir}',
                '+mephisto.blueprint.sufficient_matchups_multiplier=2',
                '+mephisto.task.task_name=acute_eval_test',
            ]
            # TODO: clean this up when Hydra has support for recursive defaults
            models = ['blender_90m_copy1', 'blender_90m_copy2']
            model_string = ','.join(models)

            # Copy over expected self-chat files
            shutil.copytree(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'self_chats'),
                os.path.join(root_dir, 'self_chats'),
            )

            # Define output structure
            outputs = {}

            # # Run Fast ACUTEs and analysis on the base task

            # Set up config
            base_task_overrides = [
                f'+mephisto.blueprint.config_path={FAST_ACUTE_TASK_DIRECTORY}/task_config/model_config.json',
                f'+mephisto.blueprint.models=\"{model_string}\"',
                '+mephisto.blueprint.model_pairs=""',
                '+mephisto.blueprint.selfchat_max_turns=6',
                '+mephisto.blueprint.task=blended_skill_talk',
                '+mephisto.blueprint.use_existing_self_chat_files=True',
            ]
            # TODO: clean this up when Hydra has support for recursive defaults
            self._set_up_config(
                blueprint_type=BASE_BLUEPRINT_TYPE,
                task_directory=ACUTE_EVAL_TASK_DIRECTORY,
                overrides=common_overrides + base_task_overrides,
            )
            self.config.mephisto.blueprint.model_pairs = None
            # TODO: hack to manually set mephisto.blueprint.model_pairs to None. Remove
            #  when Hydra releases support for recursive defaults

            # Run Fast ACUTEs
            base_runner = FastAcuteExecutor(self.config)
            base_runner.run_selfchat()
            base_runner.set_up_acute_eval()
            self.config.mephisto.blueprint = base_runner.fast_acute_args
            self._set_up_server()
            outputs['base_state'] = self._get_agent_state(task_data=self.TASK_DATA)

            # Run analysis
            base_runner.analyze_results(args=f'--mephisto-root {self.database_path}')
            outputs['base_results_path'] = base_runner.results_path

            # # Run Q-function Fast ACUTEs and analysis on the base task

            # Save the config file
            config_path = os.path.join(root_dir, 'config.json')
            config = {}
            for model in models:
                config[model] = {
                    'log_path': base_runner._get_selfchat_log_path(model),
                    'is_selfchat': True,
                }
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Set up config
            assert len(models) == 2
            q_function_overrides = common_overrides + [
                f'+mephisto.blueprint.config_path={config_path}',
                '+mephisto.blueprint.models=""',
                f'+mephisto.blueprint.model_pairs={models[0]}:{models[1]}',
            ]
            # TODO: clean this up when Hydra has support for recursive defaults
            self._set_up_config(
                blueprint_type=Q_FUNCTION_BLUEPRINT_TYPE,
                task_directory=ACUTE_EVAL_TASK_DIRECTORY,
                overrides=q_function_overrides,
            )
            self.config.mephisto.blueprint.models = None
            # TODO: hack to manually set mephisto.blueprint.models to None. Remove when
            #  Hydra releases support for recursive defaults

            # Run Fast ACUTEs
            q_function_runner = QLearningFastAcuteExecutor(self.config)
            q_function_runner.set_up_acute_eval()
            self.config.mephisto.blueprint = q_function_runner.fast_acute_args
            self._set_up_server()
            outputs['q_function_task_state'] = self._get_agent_state(
                task_data=self.TASK_DATA
            )

            # Run analysis
            q_function_runner.analyze_results(
                args=f'--mephisto-root {self.database_path}'
            )
            outputs['q_function_results_path'] = q_function_runner.results_path

            yield outputs
            # All code after this will be run upon teardown

            self._teardown()

            # Tear down temp file
            shutil.rmtree(root_dir)

        def test_base_agent_state(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):
            outputs = setup_teardown
            self._check_agent_state(
                state=outputs['base_state'], data_regression=data_regression
            )

        def test_

        def test_q_function_agent_state(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):
            outputs = setup_teardown
            self._check_agent_state(
                state=outputs['q_function_state'], data_regression=data_regression
            )

        def test_base_task(
            self,
            setup_teardown,
            data_regression: DataRegressionFixture,
            dataframe_regression: DataFrameRegressionFixture,
            file_regression: FileRegressionFixture,
        ):

            outputs = setup_teardown

            self._check_agent_state()

            self._check_analysis_outputs(
                outputs_folder=self.base_task_runner.results_path,
                save_prefix='base',
                dataframe_regression=dataframe_regression,
                file_regression=file_regression,
            )

        def test_q_function_task(
            self,
            setup_teardown,
            data_regression: DataRegressionFixture,
            dataframe_regression: DataFrameRegressionFixture,
            file_regression: FileRegressionFixture,
        ):

            outputs = setup_teardown

            self._check_agent_state()

            # Run analysis and check outputs
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


# except ImportError:
#     pass

if __name__ == "__main__":
    unittest.main()
