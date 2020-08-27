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
                task='integration_tests:nocandidate',
                model='hugging_face/dialogpt',
                add_special_tokens=True,
                add_start_token=True,
                optimizer='sgd',
                learningrate=1,
                batchsize=4,
                num_epochs=4,
                short_final_eval=True,
                validation_max_exs=12,
            )
        )

        self.assertLessEqual(valid['ppl'], 4.0)
        self.assertLessEqual(test['ppl'], 4.0)


if __name__ == '__main__':
    unittest.main()
