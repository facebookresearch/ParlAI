#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# http://google.com
# LICENSE file in the root directory of this source tree.

"""
Tests that interactive.py is behaving well.
"""

import unittest
from unittest import mock
import parlai.scripts.interactive as interactive


class FakeInput(object):
    def __init__(self, max_turns=2):
        self.max_turns = max_turns
        self.turn = 0

    def __call__(self, prompt=""):
        self.turn += 1
        if self.turn <= self.max_turns:
            return f'Turn {self.turn}'
        else:
            return '[EXIT]'


class TestInteractive(unittest.TestCase):
    def setUp(self):
        # override input() with our fake, deterministic output
        patcher = mock.patch('builtins.input', new=FakeInput())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_repeat(self):
        pp = interactive.setup_args()
        opt = pp.parse_args(['-m', 'repeat_query'], print_args=False)
        interactive.interactive(opt)


class TestInteractiveConvai2(unittest.TestCase):
    def setUp(self):
        # override input() with our fake, deterministic output
        patcher = mock.patch('builtins.input', new=FakeInput())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_repeat(self):
        pp = interactive.setup_args()
        opt = pp.parse_args(
            ['-m', 'repeat_query', '-t', 'convai2', '-dt', 'valid'], print_args=False
        )
        interactive.interactive(opt)


if __name__ == '__main__':
    unittest.main()
