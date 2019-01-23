#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from examples.eval_model import eval_model, setup_args

import ast
import unittest
import sys


class TestEvalModel(unittest.TestCase):
    """Basic tests on the eval_model.py example."""

    def test_output(self):
        """Test output of running eval_model"""
        class display_output(object):
            def __init__(self):
                self.data = []

            def write(self, s):
                self.data.append(s)

            def __str__(self):
                return "".join(self.data)

        parser = setup_args()
        parser.set_defaults(
            task='tasks.repeat:RepeatTeacher:10',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
        )

        old_out = sys.stdout
        output = display_output()
        try:
            sys.stdout = output
            opt = parser.parse_args(print_args=False)
            eval_model(opt, print_parser=parser)
        finally:
            # restore sys.stdout
            sys.stdout = old_out

        str_output = str(output)
        self.assertTrue(len(str_output) > 0, "Output is empty")

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
