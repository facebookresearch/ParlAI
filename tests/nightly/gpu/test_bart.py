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
                short_final_eval=True,
                validation_max_exs=12,
            )
        )
        self.assertLessEqual(valid['ppl'], 11.0)
        self.assertLessEqual(test['ppl'], 11.0)


if __name__ == '__main__':
    unittest.main()
