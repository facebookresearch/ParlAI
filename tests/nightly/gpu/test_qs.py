#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import subprocess
import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestQuickStart(unittest.TestCase):
    """Runs the quickstart test script"""

    def run_quickstart_sh(self):
        subprocess.call(['sh tests/test_quickstart.sh'])


if __name__ == '__main__':
    unittest.main()
