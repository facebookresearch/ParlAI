#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestDialogptModel(unittest.TestCase):
    """
    Test of DialoGPT model.

    Checks that DialoGPT gets a certain performance on the integration test task.
    """

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
