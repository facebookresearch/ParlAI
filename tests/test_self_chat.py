#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.scripts.self_chat as self_chat


class TestSelfChat(unittest.TestCase):
    def test_vanilla(self):
        self_chat.SelfChat.main(model='fixed_response', fixed_response='hi')

    def test_convai2(self):
        self_chat.SelfChat.main(
            model='fixed_response',
            fixed_response='hi',
            task='convai2',
            datatype='valid',
        )
