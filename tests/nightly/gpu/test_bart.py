#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
import parlai.utils.testing as testing_utils
from parlai.utils.testing import tempdir


@testing_utils.skipUnlessGPU
class TestBartModel(unittest.TestCase):
    """
    Test of BART model.
    """

    def test_bart(self):
        """
        Test out-of-the-box BART on repeat task.
        """
        valid, _ = testing_utils.eval_model(
            dict(task='integration_tests', model='bart', num_examples=10)
        )
        self.assertAlmostEqual(valid['ppl'].value(), 1.0, places=1)

    def test_bart_gen(self):
        """
        Test out-of-the-box BART generation.
        """
        opt = ParlaiParser(True, True).parse_args(['--model', 'bart'])
        bart = create_agent(opt)
        text = "Don't have a cow, Man!"
        obs = {"text": text, 'episode_done': True}
        bart.observe(obs)
        act = bart.act()

        self.assertEqual(act['text'], text)

    def test_bart_cache_text_vec(self):
        """
        Test BART text vec caching.
        """
        opt = ParlaiParser(True, True).parse_args(['--model', 'bart'])
        bart = create_agent(opt)

        # obs 1
        text = "Don't have a cow, Man!"
        in_obs = {'text': text, 'episode_done': True}
        out_obs = bart.observe(in_obs)
        cached_text_vec = out_obs['text_vec']
        _ = bart.act()

        # obs 2
        in_obs = {'text_vec': cached_text_vec, 'episode_done': True}
        out_obs = bart.observe(in_obs)
        cached_text_vec_2 = out_obs['text_vec']

        self.assertEqual(cached_text_vec.tolist(), cached_text_vec_2.tolist())

    @testing_utils.retry(ntries=3, log_retry=True)
    def test_bart_ft(self):
        """
        FT BART on a "reverse" task (the opposite of what it was trained to do)
        """
        with tempdir() as tmpdir:
            # test finetuning
            mf = os.path.join(tmpdir, 'model')
            valid, test = testing_utils.train_model(
                dict(
                    task='integration_tests:reverse',
                    model='bart',
                    dict_file='zoo:bart/bart_large/model.dict',
                    optimizer='sgd',
                    learningrate=1,
                    batchsize=2,
                    num_epochs=1,
                    short_final_eval=True,
                    validation_max_exs=12,
                    model_file=mf,
                    model_parallel=True,
                    fp16=True,
                )
            )
            self.assertAlmostEqual(valid['ppl'].value(), 1.0, places=1)
            self.assertAlmostEqual(test['ppl'].value(), 1.0, places=1)

            # test generation
            opt = ParlaiParser(True, True).parse_args(['--model-file', mf])
            bart = create_agent(opt)
            text = '1 2 3 4'
            obs = {'text': text, 'episode_done': True}
            bart.observe(obs)
            act = bart.act()
            self.assertEqual(act['text'], text[::-1])


if __name__ == '__main__':
    unittest.main()
