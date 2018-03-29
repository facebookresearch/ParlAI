# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from examples.train_model import TrainLoop, setup_args

import ast
import unittest
import sys


class TestTrainModel(unittest.TestCase):
    """Basic tests on the train_model.py example."""

    def test_output(self):
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
            try:
                import torch
            except ImportError:
                print('Cannot import torch, skipping test_train_model')
                return
            parser = setup_args()
            parser.set_defaults(
                model='memnn',
                task='tasks.repeat:RepeatTeacher:10',
                dict_file='/tmp/repeat',
                batchsize='1',
                validation_every_n_secs='5',
                validation_patience='2',
                embedding_size='8',
                no_cuda=True
            )
            TrainLoop(parser).train()
        finally:
            # restore sys.stdout
            sys.stdout = old_out

        str_output = str(output)

        self.assertTrue(len(str_output) > 0, "Output is empty")
        self.assertTrue("[ training... ]" in str_output,
                        "Did not reach training step")
        self.assertTrue("[ running eval: valid ]" in str_output,
                        "Did not reach validation step")
        self.assertTrue("valid:{'total': 10," in str_output,
                        "Did not complete validation")
        self.assertTrue("[ running eval: test ]" in str_output,
                        "Did not reach evaluation step")
        self.assertTrue("test:{'total': 10," in str_output,
                        "Did not complete evaluation")

        list_output = str_output.split("\n")
        for line in list_output:
            if "test:{" in line:
                score = ast.literal_eval(line.split("test:", 1)[1])
                self.assertTrue(score['accuracy'] == 1,
                                "Accuracy did not reach 1, was " + str(score['accuracy']))

if __name__ == '__main__':
    unittest.main()
