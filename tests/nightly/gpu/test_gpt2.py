#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from parlai.core.agents import create_agent


class TestGpt2(unittest.TestCase):
    def _test_batchsize(self, batchsize, add_start_token):
        utterances = [
            'Just keep swimming -',
            'I wish I knew how to quit you. -',
            "I'm going to make him an offer he can't refuse. -",
            "Toto, I've got a feeling we're not in Kansas anymore. -",
        ]
        opt = {
            'model': 'hugging_face/gpt2',
            'gpt2_size': 'small',
            'text_truncate': 16,
            'label_truncate': 8,
            'beam_min_length': 8,
            'inference': 'beam',
            'beam_size': 1,
            'add_special_tokens': True,
            'batchsize': batchsize,
            'add_start_token': add_start_token,
        }
        gpt2 = create_agent(opt)

        results_single = []
        agents = [gpt2.clone() for _ in utterances]
        for u, a in zip(utterances, agents):
            a.observe({'text': u, 'episode_done': True})
            generation = a.act()['text']
            results_single.append(generation)

        results_batched = []
        for idx in range(len(utterances) // batchsize):
            agents = [gpt2.clone() for _ in range(batchsize)]
            batch = utterances[idx * batchsize : (idx + 1) * batchsize]
            obs = []
            for i, a in enumerate(agents):
                obs.append(a.observe({'text': batch[i], 'episode_done': True}))
            generations = [x['text'] for x in gpt2.batch_act(obs)]
            results_batched += generations

        assert results_single == results_batched

    def test_start_token(self):
        with self.assertRaises(RuntimeError):
            create_agent(
                {
                    'model': 'hugging_face/gpt2',
                    'add_special_tokens': False,
                    'add_start_token': True,
                }
            )

    def test_batchsize(self):
        """
        Ensures gpt2 provides the same generation results regardless of batchsize.
        """
        for add_start_token in [True, False]:
            for batchsize in [1, 2, 4]:
                with self.subTest(
                    f'test_batchsize with bs={batchsize} and ast={add_start_token}'
                ):
                    self._test_batchsize(batchsize, add_start_token)

    def test_nospecialtok(self):
        with self.assertRaises(RuntimeError):
            create_agent(
                {
                    'model': 'hugging_face/gpt2',
                    'add_special_tokens': False,
                    'batchsize': 2,
                }
            )

        opt = {
            'model': 'hugging_face/gpt2',
            'gpt2_size': 'small',
            'text_truncate': 16,
            'label_truncate': 8,
            'beam_min_length': 8,
            'inference': 'beam',
            'beam_size': 1,
            'batchsize': 1,
            'add_special_tokens': False,
        }
        gpt2 = create_agent(opt)
        gpt2.observe({'text': 'My name is', 'episode_done': True})
        response = gpt2.act()
        assert response['text'] == " John. I'm a man of"
