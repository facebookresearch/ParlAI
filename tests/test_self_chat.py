#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from parlai.scripts.self_chat import SelfChat


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
            model_file='zoo:tutorial_transformer_generator/model',
            seed_messages_from_task=True,
        )

    def test_no_plain_teacher(self):
        from parlai.scripts.display_data import DisplayData

        with self.assertRaises(RuntimeError):
            DisplayData.main(task='self_chat')
