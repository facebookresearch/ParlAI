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
import os
import parlai.scripts.interactive as interactive
import parlai.utils.conversations as conversations
import parlai.utils.testing as testing_utils


class FakeInput(object):
    def __init__(self, max_turns=2, max_episodes=1, clean_exit=True):
        self.max_turns = max_turns
        self.turn = 0
        self.max_episodes = max_episodes
        self.episode = 1
        self.clean_exit = clean_exit

    def __call__(self, prompt=""):
        self.turn += 1
        if self.turn <= self.max_turns:
            return f'Turn {self.turn}'

        self.turn = 0
        self.episode += 1
        if self.episode <= self.max_episodes:
            return '[DONE]'
        elif self.clean_exit:
            return '[EXIT]'
        else:
            raise EOFError


class TestInteractive(unittest.TestCase):
    def setUp(self):
        # override input() with our fake, deterministic output
        patcher = mock.patch('builtins.input', new=FakeInput())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_repeat(self):
        pp = interactive.setup_args()
        opt = pp.parse_args(['-m', 'repeat_query'])
        interactive.interactive(opt)


class TestInteractiveConvai2(unittest.TestCase):
    def setUp(self):
        # override input() with our fake, deterministic output
        patcher = mock.patch('builtins.input', new=FakeInput())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_repeat(self):
        pp = interactive.setup_args()
        opt = pp.parse_args(['-m', 'repeat_query', '-t', 'convai2', '-dt', 'valid'])
        interactive.interactive(opt)


class TestInteractiveLogging(unittest.TestCase):
    def test_repeat(self):
        for clean in (True, False):
            with self.subTest(clean=clean), testing_utils.tempdir() as tmpdir:
                fake_input = FakeInput(max_episodes=2, clean_exit=clean)
                with mock.patch('builtins.input', new=fake_input):
                    self._run_test_repeat(tmpdir, fake_input)

    def _run_test_repeat(self, tmpdir: str, fake_input: FakeInput):
        outfile = os.path.join(tmpdir, 'log.jsonl')
        pp = interactive.setup_args()
        opt = pp.parse_args(['-m', 'repeat_query', '--outfile', outfile])
        interactive.interactive(opt)

        log = conversations.Conversations(outfile)
        self.assertEqual(len(log), fake_input.max_episodes)
        for entry in log:
            self.assertEqual(len(entry), 2 * fake_input.max_turns)


if __name__ == '__main__':
    unittest.main()
