# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from examples.eval_model import eval_model
from parlai.core.params import ParlaiParser

import ast
import unittest
import sys


class TestEvalModel(unittest.TestCase):
    """Basic tests on the eval_model.py example."""

    args = [
        '--task', '#moviedd-reddit',
        '--datatype', 'valid',
    ]

    parser = ParlaiParser()
    parser.set_defaults(datatype='valid')
    opt = parser.parse_args(args, print_args=False)
    opt['model'] = 'repeat_label'
    opt['num_examples'] = 5
    opt['display_examples'] = False

    def test_output(self):
        """Test output of running eval_model"""
        class display_output(object):
            def __init__(self):
                self.data = []

            def write(self, s):
                self.data.append(s)

            def __str__(self):
                return "".join(self.data)

        old_out = sys.stdout
        output = display_output()
        try:
            sys.stdout = output
            eval_model(self.opt, self.parser, printargs=False)
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
            self.assertTrue(score['total'] == i,
                            "Total is incorrect")
            # accuracy should be one
            self.assertTrue(score['accuracy'] == 1,
                            "accuracy != 1")

if __name__ == '__main__':
    unittest.main()
