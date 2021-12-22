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
from parlai.scripts.interactive import Interactive
from parlai.scripts.safe_interactive import SafeInteractive
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
        Interactive.main(model='repeat_query')

    def test_safe_interactive(self):
        SafeInteractive.main(model='repeat_query')


class TestInteractiveConvai2(unittest.TestCase):
    def setUp(self):
        # override input() with our fake, deterministic output
        patcher = mock.patch('builtins.input', new=FakeInput())
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_repeat(self):
        Interactive.main(model='repeat_query', task='convai2', datatype='valid')


class TestInteractiveLogging(unittest.TestCase):
    def test_repeat(self):
        for clean in (True, False):
            with self.subTest(clean=clean), testing_utils.tempdir() as tmpdir:
                fake_input = FakeInput(max_episodes=2, clean_exit=clean)
                with mock.patch('builtins.input', new=fake_input):
                    self._run_test_repeat(tmpdir, fake_input)

    def _run_test_repeat(self, tmpdir: str, fake_input: FakeInput):
        outfile = os.path.join(tmpdir, 'log.jsonl')
        Interactive.main(model='repeat_query', outfile=outfile)

        log = conversations.Conversations(outfile)
        self.assertEqual(len(log), fake_input.max_episodes)
        for entry in log:
            self.assertEqual(len(entry), 2 * fake_input.max_turns)


class TestInteractiveWeb(unittest.TestCase):
    def test_iweb(self, task: str = None):
        import threading
        import random
        import requests
        import json
        import parlai.scripts.interactive_web as iweb

        port = random.randint(30000, 40000)
        kwargs = {'model': 'repeat_query', 'port': port}
        if task:
            kwargs['task'] = task
        thread = threading.Thread(
            target=iweb.InteractiveWeb.main, kwargs=kwargs, daemon=True
        )
        thread.start()
        iweb.wait()

        r = requests.get(f'http://localhost:{port}/')
        assert '<html>' in r.text

        r = requests.post(f'http://localhost:{port}/interact', data='This is a test')
        assert r.status_code == 200
        response = json.loads(r.text)
        assert 'text' in response
        assert response['text'] == 'This is a test'

        r = requests.post(f'http://localhost:{port}/reset')
        assert r.status_code == 200
        response = json.loads(r.text)
        assert response == {}

        r = requests.get(f'http://localhost:{port}/bad')
        assert r.status_code == 500

        r = requests.post(f'http://localhost:{port}/bad')
        assert r.status_code == 500

        iweb.shutdown()

    def test_iweb_task(self):
        self.test_iweb(task='convai2')


class TestProfileInteractive(unittest.TestCase):
    def test_profile_interactive(self):
        from parlai.scripts.profile_interactive import ProfileInteractive

        fake_input = FakeInput(max_episodes=2)
        with mock.patch('builtins.input', new=fake_input):
            ProfileInteractive.main(model='repeat_query')


if __name__ == '__main__':
    unittest.main()
