#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import unittest

import parlai.utils.testing as testing_utils
from parlai.utils.conversations import Conversations
from parlai.core.params import ParlaiParser


class TestConversations(unittest.TestCase):
    """
    Tests Conversations utilities.
    """

    def setUp(self):
        self.datapath = ParlaiParser().parse_args([], print_args=False)['datapath']
        self.datapath = os.path.join(self.datapath, 'tmp_conversations')
        os.makedirs(self.datapath, exist_ok=True)

    def test_conversations(self):
        act_list = [
            [
                [
                    {'id': 'Emily', 'text': 'Hello, do you like this test?'},
                    {'id': 'Stephen', 'text': 'Why yes! I love this test!'},
                ],
                [
                    {'id': 'Emily', 'text': 'So will you stamp this diff?'},
                    {'id': 'Stephen', 'text': 'Yes, I will do it right now!'},
                ],
            ],
            [
                [
                    {
                        'id': 'A',
                        'text': 'Somebody once told me the world is gonna roll me',
                    },
                    {'id': 'B', 'text': 'I aint the sharpest tool in the shed'},
                ],
                [
                    {
                        'id': 'A',
                        'text': 'She was looking kind of dumb with her finger and her thumb',
                    },
                    {'id': 'B', 'text': 'In the shape of an L on her forehead'},
                ],
            ],
        ]
        self.opt = {
            'A': 'B',
            'C': 'D',
            'E': 'F',
        }

        self.convo_datapath = os.path.join(self.datapath, 'convo1')
        Conversations.save_conversations(
            act_list,
            self.convo_datapath,
            self.opt,
            self_chat=False,
            other_info='Blah blah blah',
        )
        assert os.path.exists(self.convo_datapath + '.jsonl')
        assert os.path.exists(self.convo_datapath + '.metadata')

        convos = Conversations(self.convo_datapath + '.jsonl')

        # test conversations loaded
        self.assertEqual(len(convos), 2)

        # test speakers saved
        speakers = {'Stephen', 'Emily', 'A', 'B'}
        self.assertEqual(set(convos.metadata.speakers), speakers)

        # test opt saved
        for x in ['A', 'C', 'E']:
            self.assertEqual(
                self.opt[x], convos.metadata.opt[x],
            )

        # test kwargs
        self.assertEqual({'other_info': 'Blah blah blah'}, convos.metadata.extra_data)

        # test reading conversations
        with testing_utils.capture_output() as out:
            convos.read_conv_idx(0)
        str_version = (
            'Emily: Hello, do you like this test?\n'
            'Stephen: Why yes! I love this test!\n'
            'Emily: So will you stamp this diff?\n'
            'Stephen: Yes, I will do it right now!\n'
        )
        self.assertIn(str_version, out.getvalue())

        # test getting a specific turn
        first = convos[0]  # Conversation
        self.assertEqual(first[0].id, 'Emily')
        self.assertEqual(first[3].text, 'Yes, I will do it right now!')

    def tearDown(self):
        # remove conversations
        shutil.rmtree(self.datapath)


if __name__ == '__main__':
    unittest.main()
