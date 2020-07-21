#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestBartModel(unittest.TestCase):
    """
    Test of BART model.

    Checks that BART can be trained on ~ 1k samples of convai2
    """

    @testing_utils.retry(ntries=3, log_retry=True)
    def test_bart(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:nocandidate',
                model='bart',
                dict_file='zoo:bart/bart_large/model.dict',
                optimizer='sgd',
                learningrate=1,
                batchsize=4,
                num_epochs=1,
            )
        )
        self.assertAlmostEqual(test['ppl'], 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
