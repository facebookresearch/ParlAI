#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message

import unittest


class TestUtils(unittest.TestCase):
    def test_message(self):
        message = Message()
        message['text'] = 'lol'
        err = None
        try:
            message['text'] = 'rofl'
        except RuntimeError as e:
            err = e
        assert err is not None, 'Message allowed override'
        message_copy = message.copy()
        assert type(message_copy) == Message, 'Message did not copy properly'
