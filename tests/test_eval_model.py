#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from examples.eval_model import setup_args

import ast
import unittest
import parlai.core.testing_utils as testing_utils


class TestEvalModel(unittest.TestCase):
    """Basic tests on the eval_model.py example."""

    def test_output(self):
        """Test output of running eval_model"""
        parser = setup_args()
        parser.set_defaults(
            task='tasks.repeat:RepeatTeacher:10',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
        )

        opt = parser.parse_args(print_args=False)
        str_output, valid, test = testing_utils.eval_model(opt)
        self.assertGreater(len(str_output), 0, "Output is empty")

        # decode the output
        scores = str_output.split("\n---\n")
        for i in range(1, len(scores)):
            score = ast.literal_eval(scores[i])
            # check totals
            self.assertTrue(score['exs'] == i,
                            "Total is incorrect")
            # accuracy should be one
            self.assertTrue(score['accuracy'] == 1,
                            "accuracy != 1")


if __name__ == '__main__':
    unittest.main()
