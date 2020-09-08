#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.scripts.self_chat as self_chat
import parlai.scripts.convo_render as convo_render


class TestConvoRender(unittest.TestCase):
    def test_convo_render(self):
        """
        Test convo render by creating a self-chat, saving it to file and render it to
        html.
        """
        self_chat_pp = self_chat.setup_args()
        self_chat_opt = self_chat_pp.parse_args(
            [
                '-m',
                'fixed_response',
                '--fixed-response',
                'Hey there',
                '--save-format',
                'conversations',
                '--outfile',
                'self_chat_output',
            ]
        )
        self_chat.self_chat(self_chat_opt)

        convo_render_pp = convo_render.setup_args()
        convo_render_opt = convo_render_pp.parse_args(
            ['-i', 'self_chat_output.jsonl', '-o', 'self_chat_output.html']
        )
        convo_render.render_convo(convo_render_opt)
