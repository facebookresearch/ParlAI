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

    @testing_utils.retry(ntries=3)
    def test_generation(self):
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
                inference="greedy",
                numlayers=1,
                embeddingsize=16,
                hiddensize=32,
                batchsize=8,
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
                embeddingsize=16,
                hiddensize=32,
                batchsize=8,
                inference="beam",
                beam_size=5,
            )
        )
        self.assertGreater(valid["accuracy"], 0.95)
        self.assertGreater(test["accuracy"], 0.95)

    def test_badinput(self):
        """
        Ensures model doesn't crash on malformed inputs.
        """
        testing_utils.train_model(
            dict(
                task="integration_tests:bad_example",
                model="hred",
                learningrate=1,
                batchsize=10,
                datatype="train:ordered:stream",
                num_epochs=1,
                embeddingsize=16,
                hiddensize=16,
                inference="greedy",
            )
        )


if __name__ == "__main__":
    unittest.main()

