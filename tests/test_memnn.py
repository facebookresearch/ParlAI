#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import unittest
from test_torch_agent import SKIP_TESTS, MockDict
from parlai.agents.memnn.memnn import MemnnAgent


class MockDict(MockDict):
    # also allow "time features"
    def __setitem__(self, key, value):
        self.idx += 1

    def __len__(self):
        return self.idx


class MemnnAgent(MemnnAgent):
    @staticmethod
    def dictionary_class():
        return MockDict


def get_agent(**kwargs):
    """Return opt-initialized agent.

    :param kwargs: any kwargs you want to set using parser.set_params(**kwargs)
    """
    if 'no_cuda' not in kwargs:
        kwargs['no_cuda'] = True
    from parlai.core.params import ParlaiParser
    parser = ParlaiParser()
    MemnnAgent.add_cmdline_args(parser)
    parser.set_params(**kwargs)
    opt = parser.parse_args(print_args=False)
    return MemnnAgent(opt)


class TestMemnn(unittest.TestCase):
    @unittest.skipIf(SKIP_TESTS, "Torch not installed.")
    def test_batchify(self):
        agent = get_agent(rank_candidates=True)
        obs_labs = [
            {'text': 'It\'s only a flesh wound.',
             'labels': ['Yield!']},
            {'text': 'The needs of the many outweigh...',
             'labels': ['The needs of the few.']},
            {'text': 'Hello there.',
             'labels': ['General Kenobi.']},
        ]

        obs_vecs = [agent.vectorize(o) for o in obs_labs]
        batch = agent.batchify(obs_vecs)
        self.assertIsNotNone(batch.memory_vecs)

    @unittest.skipIf(SKIP_TESTS, "Torch not installed.")
    def test_vectorize(self):
        agent = get_agent()
        obs = {
            'text': 'Hello.\nMy name is Inogo Montoya.\n'
                    'You killed my father.\nPrepare to die.',
        }
        out = agent.vectorize(obs)
        self.assertEqual(out['text_vec'].tolist(), [1, 2, 3])  # last line
        self.assertEqual([m.tolist() for m in out['memory_vecs']],
                         [[1], [1, 2, 3, 4, 5], [1, 2, 3, 4]])
        # check cache
        out_again = agent.vectorize(obs)
        self.assertIs(out['text_vec'], out_again['text_vec'])
        self.assertIs(out['memory_vecs'], out_again['memory_vecs'])
        self.assertEqual(out['text_vec'].tolist(), [1, 2, 3])
        self.assertEqual([m.tolist() for m in out['memory_vecs']],
                         [[1], [1, 2, 3, 4, 5], [1, 2, 3, 4]])
        # next: should truncate cached result
        prev_vec = out['text_vec']
        prev_mem = out['memory_vecs']
        out_again = agent.vectorize(out, truncate=1)
        self.assertIsNot(prev_vec, out_again['text_vec'])
        self.assertEqual(out['text_vec'].tolist(), [1])
        self.assertIsNot(prev_mem, out_again['memory_vecs'])
        for i in range(len(prev_mem)):
            if len(prev_mem[i]) > 1:
                # if truncated, different tensor
                self.assertIsNot(prev_mem[i], out_again['memory_vecs'][i])
            else:
                # otherwise should still be the same one
                self.assertIs(prev_mem[i], out_again['memory_vecs'][i])
        self.assertEqual([m.tolist() for m in out['memory_vecs']],
                         [[1], [1], [1]])
