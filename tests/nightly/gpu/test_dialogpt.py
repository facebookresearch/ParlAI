#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils
from parlai.core.agents import create_agent


@testing_utils.skipUnlessGPU
class TestDialogptModel(unittest.TestCase):
    """
    Test of DialoGPT model.
    """

    def _test_batchsize(self, batchsize, add_start_token):
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
            'add_special_tokens': True,
            'batchsize': batchsize,
            'add_start_token': add_start_token,
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

        assert results_single == results_batched

    def test_batchsize(self):
        """
        Ensures dialogpt provides the same generation results regardless of batchsize.
        """
        # Test throwing the RuntimeError with add_special_tokens = False and batchsize > 1
        with self.assertRaises(RuntimeError):
            create_agent(
                {
                    'model': 'hugging_face/dialogpt',
                    'add_special_tokens': False,
                    'batchsize': 2,
                }
            )

        for batchsize in [1, 2, 4]:
            for add_start_token in [True, False]:
                with self.subTest(
                    f'test_batchsize with bs={batchsize} and add_start_token={add_start_token}'
                ):
                    self._test_batchsize(batchsize, add_start_token)

    def test_start_token(self):
        """
        Test RuntimeError is thrown when add_start_token = True and yet add_special_tokens = False
        """
        with self.assertRaises(RuntimeError):
            create_agent(
                {
                    'model': 'hugging_face/dialogpt',
                    'add_special_tokens': False,
                    'add_start_token': True,
                }
            )

    def test_nospecialtok(self):
        """
        Test generation consistency for off-the-shelf dialogpt models.
        """
        test_cases = [
            ("What a nice weather!", "I'm in the UK and it's raining here."),
            ("Nice to meet you!", "Hello! I'm from the future!"),
        ]
        opt = {
            'model': 'hugging_face/dialogpt',
            'gpt2_size': 'small',
            'text_truncate': 100,
            'label_truncate': 20,
            'beam_min_length': 1,
            'inference': 'beam',
            'beam_size': 1,
            'add_special_tokens': False,
            'batchsize': 1,
        }
        dialogpt = create_agent(opt)
        for text, label in test_cases:
            dialogpt.observe({'text': text, 'episode_done': True})
            response = dialogpt.act()
            assert response['text'] == label

    def test_dialogpt(self):
        """
        Checks that DialoGPT gets a certain performance on the integration test task.
        """
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:overfit',
                model='hugging_face/dialogpt',
                add_special_tokens=True,
                add_start_token=True,
                optimizer='adam',
                learningrate=1e-3,
                batchsize=1,
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
