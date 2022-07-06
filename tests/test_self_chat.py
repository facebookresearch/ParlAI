#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tempfile import NamedTemporaryFile
import unittest

from parlai.scripts.self_chat import SelfChat
import parlai.utils.testing as testing_utils


class TestSelfChat(unittest.TestCase):
    def test_vanilla(self):
        SelfChat.main(model='fixed_response', fixed_response='hi')

    def test_convai2(self):
        SelfChat.main(
            task='convai2', model='fixed_response', fixed_response='hi', dt='valid'
        )

    def test_ed(self):
        SelfChat.main(
            task='empathetic_dialogues',
            model='fixed_response',
            fixed_response='hi',
            seed_messages_from_task=True,
        )

    def test_no_plain_teacher(self):
        from parlai.scripts.display_data import DisplayData

        with self.assertRaises(RuntimeError):
            DisplayData.main(task='self_chat')

    def test_seed_messages_from_file(self):
        with testing_utils.capture_output() as output:
            with NamedTemporaryFile() as tmpfile:
                tmpfile.write(b'howdy\nunique message')
                tmpfile.seek(0)
                SelfChat.main(
                    model='fixed_response',
                    fixed_response='hi',
                    seed_messages_from_file=tmpfile.name,
                    num_self_chats=10,
                    selfchat_max_turns=2,
                )
            output = output.getvalue()
        assert 'howdy' in output
        assert 'unique message' in output
