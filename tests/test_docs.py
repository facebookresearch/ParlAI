#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests that the docs build.
"""

import os
import unittest
import parlai
import subprocess
try:
    import sphinx  # noqa: F401
    SKIP_TEST = False
except ImportError:
    SKIP_TEST = True

# find the source directory for parlai
SOURCE_ROOT = os.path.dirname(os.path.dirname(parlai.__file__))
DOCS_DIR = os.path.join(SOURCE_ROOT, "docs")


class TestDocs(unittest.TestCase):
    @unittest.skipIf(SKIP_TEST, "Sphinx not installed.")
    def test_docs_build(self):
        call = subprocess.run(
            ["make", "html"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=DOCS_DIR,
            universal_newlines=True,
        )
        self.assertEqual(
            call.returncode, 0,
            'Failed to compile docs:\n{}'.format(call.stderr)
        )


if __name__ == '__main__':
    unittest.main()
