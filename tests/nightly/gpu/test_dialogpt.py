#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils
from parlai.core.agents import create_agent
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


@testing_utils.skipUnlessGPU
class TestDialogptModel(unittest.TestCase):
    """
    Test of DialoGPT model.

    Checks that DialoGPT gets a certain performance on the integration test task.
    """

    def _test_batchsize(self, batchsize, add_special_tokens):
        utterances = [
            'How is your day so far?',
            'I hope you you have a good day.',
            "Nice to meet you. My name is John. ",
            "I've got a feeling we're not in Kansas anymore.",
        ]
        opt = {
            'model': 'hugging_face/dialogpt',
            'gpt2_size': 'small',
            'text_truncate': 100,
            'label_truncate': 20,
            'beam_min_length': 1,
            'inference': 'beam',
            'beam_size': 1,
            'add_special_tokens': add_special_tokens,
            'batchsize': batchsize,
            'add_start_token': False,
        }
        dialogpt = create_agent(opt)

        results_single = []
        agents = [dialogpt.clone() for _ in utterances]
        for u, a in zip(utterances, agents):
            a.observe({'text': u, 'episode_done': True})
            generation = a.act()['text']
            results_single.append(generation)

        results_batched = []
        for idx in range(len(utterances) // batchsize):
            agents = [dialogpt.clone() for _ in range(batchsize)]
            batch = utterances[idx * batchsize : (idx + 1) * batchsize]
            obs = []
            for i, a in enumerate(agents):
                obs.append(a.observe({'text': batch[i], 'episode_done': True}))
            generations = [x['text'] for x in dialogpt.batch_act(obs)]
            results_batched += generations

        print(f'results_single = {results_single}')
        print(f'results_batched = {results_batched}')
        assert results_single == results_batched

    def test_batchsize(self):
        """
        Ensures dialogpt provides the same generation results regardless of batchsize.
        """
        for batchsize in [2, 2, 4, 2]:
            for add_special_tokens in [True]:
                if batchsize > 1 and not add_special_tokens:
                    continue
                with self.subTest(
                    f'test_batchsize with bs={batchsize} and add_special_token={add_special_tokens}'
                ):
                    print(
                        f'_____________test_batchsize with bs={batchsize} and add_special_token={add_special_tokens}'
                    )
                    self._test_batchsize(batchsize, add_special_tokens)

    @testing_utils.retry(ntries=3, log_retry=True)
    def test_dialogpt(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:overfit',
                model='hugging_face/dialogpt',
                add_special_tokens=True,
                add_start_token=True,
                optimizer='adam',
                learningrate=1e-3,
                batchsize=4,
                num_epochs=100,
                validation_every_n_epochs=5,
                validation_metric='ppl',
                short_final_eval=True,
                skip_generation=True,
            )
        )

        self.assertLessEqual(valid['ppl'], 4.0)
        self.assertLessEqual(test['ppl'], 4.0)


if __name__ == '__main__':
    unittest.main()
