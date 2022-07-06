#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for general checks of code quality.
"""

import pytest
import glob
import unittest
import os
import parlai.utils.testing as testing_utils
import parlai.opt_presets as opt_presets


@pytest.mark.nofbcode
class TestInit(unittest.TestCase):
    """
    Make sure all python packages have init files.
    """

    def test_init_everywhere(self):
        for folder_path in testing_utils.git_ls_dirs('parlai'):
            excluded_folders = [
                'conf',
                'frontend',
                'mturk',
                'task_config',
                'webapp',
                'opt_presets',
            ]
            # conf: contains YAML files for Hydra
            # frontend, mturk, webapp: contains frontend code for crowdsourcing tasks
            # task_config: contains JSONs, HTML files, etc. for MTurk/Mephisto tasks
            if any(folder_name in folder_path for folder_name in excluded_folders):
                continue
            if folder_path.endswith("test") and folder_path.startswith("parlai/tasks/"):
                # yml regression files in parlai/tasks/X/test/
                continue
            self.assertIn(
                '__init__.py',
                os.listdir(folder_path),
                '{} does not contain __init__.py'.format(folder_path),
            )


@pytest.mark.nofbcode
class TestOptPresets(unittest.TestCase):
    """
    Ensure all opt presets contain descriptions.
    """

    def test_opt_preset_docs(self):
        from parlai.opt_presets.docs import PRESET_DESCRIPTIONS

        folder = os.path.dirname(opt_presets.__file__)
        has_file = set(x[len(folder) + 1 : -4] for x in glob.glob(f'{folder}/*/*.opt'))
        has_docs = set(PRESET_DESCRIPTIONS.keys())

        file_no_docs = has_file - has_docs
        if file_no_docs:
            raise AssertionError(
                "The following opt presets have files but no documentation: "
                f"{', '.join(file_no_docs)}"
            )
        docs_no_file = has_docs - has_file
        if docs_no_file:
            raise AssertionError(
                "The following opt presets have documentation but no files: "
                f"{', '.join(docs_no_file)}"
            )


if __name__ == '__main__':
    unittest.main()
