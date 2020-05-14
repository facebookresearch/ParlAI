#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.scripts.self_chat as self_chat


class TestSelfChat(unittest.TestCase):
    def test_vanilla(self):
        pp = self_chat.setup_args()
        opt = pp.parse_args(['-m', 'fixed_response', '--fixed-response', 'hi'])
        self_chat.self_chat(opt)

    def test_convai2(self):
        pp = self_chat.setup_args()
        opt = pp.parse_args(
            [
                '-m',
                'fixed_response',
                '--fixed-response',
                'hi',
                '-t',
                'convai2',
                '-dt',
                'valid',
            ]
        )
        self_chat.self_chat(opt)
