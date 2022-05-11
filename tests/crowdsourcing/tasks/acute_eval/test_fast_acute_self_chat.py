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

    from parlai.crowdsourcing.tasks.acute_eval.fast_eval import (
        FastAcuteExecutor,
        FAST_ACUTE_CONFIG_NAME,
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
            test_overrides = ['mephisto.blueprint.use_existing_self_chat_files=True']
            self._set_up_config(
                task_directory=self.TASK_DIRECTORY,
                overrides=self._get_common_overrides(root_dir) + test_overrides,
                config_name=FAST_ACUTE_CONFIG_NAME,
            )

            # Run Fast ACUTEs
            try:
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
            finally:
                # All code after this will be run upon teardown
                self._teardown()
                # Tear down temp file
                shutil.rmtree(root_dir)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
