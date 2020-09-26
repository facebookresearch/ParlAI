#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import parlai.utils.testing as testing_utils


class TestHred(unittest.TestCase):
    """
    Checks that Hred can learn some very basic tasks.
    """

    def test_generation(self):
        valid, test = testing_utils.train_model(
            dict(
                task="integration_tests:overfit",
                model="hred",
                optimizer='sgd',
                momentum=0.9,
                learningrate=1.0,
                lr_scheduler='none',
                batchsize=4,
                num_epochs=100,
                validation_every_n_epochs=4,
                validation_metric='ppl',
                embeddingsize=16,
                hiddensize=32,
                numlayers=1,
                dropout=0.0,
                skip_generation=True,
            )
        )
        self.assertLess(valid["ppl"], 2)

    @testing_utils.retry(ntries=3)
    def test_greedy(self):
        """
        Test a simple multiturn task.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task="integration_tests:multiturn_candidate",
                model="hred",
                model_file="zoo:unittest/hred_model/model",
                dict_file="zoo:unittest/hred_model/model.dict",
                skip_generation=False,
                batchsize=32,
            )
        )

        self.assertLess(valid["ppl"], 1.2)
        self.assertLess(test["ppl"], 1.2)

    @testing_utils.retry(ntries=3)
    def test_beamsearch(self):
        """
        Ensures beam search can generate the correct response.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task="integration_tests:multiturn_candidate",
                model="hred",
                model_file="zoo:unittest/hred_model/model",
                dict_file="zoo:unittest/hred_model/model.dict",
                skip_generation=False,
                numlayers=1,
                batchsize=8,
                inference="beam",
                beam_size=5,
                num_examples=20,
            )
        )
        self.assertGreater(valid["accuracy"], 0.95)
        self.assertGreater(test["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
