#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for general checks of code quality.
"""

import unittest
import os
import parlai.utils.testing as testing_utils


class TestInit(unittest.TestCase):
    """
    Make sure all python packages have init files.
    """

    def test_init_everywhere(self):
        for folder in testing_utils.git_ls_dirs('parlai'):
            if 'mturk' in folder:
                continue
            self.assertIn(
                '__init__.py',
                os.listdir(folder),
                '{} does not contain __init__.py'.format(folder),
            )


if __name__ == '__main__':
    unittest.main()
