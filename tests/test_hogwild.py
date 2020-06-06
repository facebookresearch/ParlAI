#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import parlai.utils.testing as testing_utils

NUM_EXS = 100
# ideally we want one choice which is a nice modulo with NUM_EXS, and one that isn't
NUM_THREADS_CHOICES = [2, 8]
BATCHSIZE_CHOICES = [1, 8]


@testing_utils.skipIfGPU
class TestHogwild(unittest.TestCase):
    """
    Check that hogwild is doing the right number of examples.
    """

    def test_hogwild_train(self):
        """
        Test the trainer eval with numthreads > 1 and batchsize in [1,2,3].
        """
        opt = dict(
            task='integration_tests:repeat:{}'.format(1),
            evaltask='integration_tests:repeat:{}'.format(NUM_EXS),
            model='repeat_label',
            display_examples=False,
            num_epochs=10,
        )
        for nt in NUM_THREADS_CHOICES:
            for bs in BATCHSIZE_CHOICES:
                opt['num_threads'] = nt
                opt['batchsize'] = bs

                valid, test = testing_utils.train_model(opt)
                self.assertEqual(valid['exs'], NUM_EXS)
                self.assertEqual(test['exs'], NUM_EXS)

    def test_hogwild_eval(self):
        """
        Test eval with numthreads > 1 and batchsize in [1,2,3].
        """
        opt = dict(
            task='integration_tests:repeat:{}'.format(NUM_EXS), model='repeat_label'
        )
        for nt in NUM_THREADS_CHOICES:
            for bs in BATCHSIZE_CHOICES:
                opt['num_threads'] = nt
                opt['batchsize'] = bs

                valid, test = testing_utils.eval_model(opt)
                self.assertEqual(valid['exs'], NUM_EXS)
                self.assertEqual(test['exs'], NUM_EXS)


if __name__ == '__main__':
    unittest.main()
