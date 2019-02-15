#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for general checks of code quality.
"""

import unittest
import os
import re
import parlai.core.testing_utils as testing_utils

FILENAME_EXTENSIONS = r'.*\.(rst|py|sh|js|css)$'
WHITELIST_PHRASES = [
    'Moscow Institute of Physics and Technology.',
    'https://github.com/fartashf/vsepp'
]
COPYRIGHT = [
    "Copyright (c) Facebook, Inc. and its affiliates.",
    "This source code is licensed under the MIT license found in the",
    "LICENSE file in the root directory of this source tree.",
]


class TestCopyright(unittest.TestCase):
    """Make sure all files have the right copyright."""

    def test_copyright(self):
        for fn in testing_utils.git_ls_files():
            # only check source files
            if not re.match(FILENAME_EXTENSIONS, fn):
                continue

            with open(fn, 'r') as f:
                src = f.read(512)  # only need the beginning

            if not src.strip():
                # skip empty files
                continue

            if any(wl in src for wl in WHITELIST_PHRASES):
                # skip a few things we don't have the copyright on
                continue

            for i, msg in enumerate(COPYRIGHT):
                if "mlb_vqa" in fn and i < 2:
                    # very special exception for mlb_vqa
                    continue

                self.assertTrue(
                    msg in src,
                    '{} missing copyright "{}"'.format(fn, msg)
                )


class TestInit(unittest.TestCase):
    """Make sure all python packages have init files."""
    def test_init_everywhere(self):
        for folder in testing_utils.git_ls_dirs('parlai'):
            if 'mturk' in folder:
                continue
            self.assertIn(
                '__init__.py',
                os.listdir(folder),
                '{} does not contain __init__.py'.format(folder)
            )


if __name__ == '__main__':
    unittest.main()
