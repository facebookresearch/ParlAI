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

import pytest


try:

    from parlai.crowdsourcing.tasks.acute_eval.fast_eval import FastAcuteExecutor
    from parlai.crowdsourcing.tasks.acute_eval.fast_acute_blueprint import (
        FAST_ACUTE_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.acute_eval.util import AbstractFastAcuteTest

    class TestFastAcuteSelfChat(AbstractFastAcuteTest):
        """
        Test the Fast ACUTE crowdsourcing task with model self-chats.
        """

        @pytest.fixture(scope="module")
        def setup_teardown(self):
            """
            Call code to set up and tear down tests.

            Run this only once because we'll be running all Fast ACUTE code before
            checking any results.
            """

            self._setup()

            # Set up common temp directory
            root_dir = tempfile.mkdtemp()

            # Copy over expected self-chat files
            shutil.copytree(
                os.path.join(self.TASK_DIRECTORY, 'task_config', 'self_chats'),
                os.path.join(root_dir, 'self_chats'),
            )

            # Define output structure
            outputs = {}

            # Set up config
            test_overrides = [
                f'+mephisto.blueprint.config_path={self.TASK_DIRECTORY}/task_config/model_config_self_chat.json',
                f'+mephisto.blueprint.models=\"{self.MODEL_STRING}\"',
                '+mephisto.blueprint.model_pairs=""',
                '+mephisto.blueprint.selfchat_max_turns=6',
                '+mephisto.blueprint.use_existing_self_chat_files=True',
            ]
            # TODO: clean this up when Hydra has support for recursive defaults
            self._set_up_config(
                blueprint_type=FAST_ACUTE_BLUEPRINT_TYPE,
                task_directory=self.TASK_DIRECTORY,
                overrides=self._get_common_overrides(root_dir) + test_overrides,
            )
            self.config.mephisto.blueprint.model_pairs = None
            # TODO: hack to manually set mephisto.blueprint.model_pairs to None. Remove
            #  when Hydra releases support for recursive defaults

            # Run Fast ACUTEs
            runner = FastAcuteExecutor(self.config)
            runner.compile_chat_logs()
            runner.set_up_acute_eval()
            self.config.mephisto.blueprint = runner.fast_acute_args
            self._set_up_server()
            outputs['state'] = self._get_agent_state(task_data=self.TASK_DATA)

            # Run analysis
            runner.analyze_results(args=f'--mephisto-root {self.database_path}')
            outputs['results_folder'] = runner.results_path

            yield outputs
            # All code after this will be run upon teardown

            self._teardown()

            # Tear down temp file
            shutil.rmtree(root_dir)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
