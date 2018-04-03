# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from examples.train_model import TrainLoop, setup_args
from examples.eval_model import eval_model

import ast
import unittest
import sys


class display_output(object):
    def __init__(self):
        self.data = []

    def write(self, s):
        self.data.append(s)

    def __str__(self):
        return "".join(self.data)


class TestHogwild(unittest.TestCase):
    """Check that hogwild is doing the right number of examples."""

    def test_hogwild_train(self):
        pass

    def test_hogwild_eval(self):
        parser = setup_args()
        NUM_EXS = 100
        parser.set_defaults(
            task='tasks.repeat:RepeatTeacher:{}'.format(NUM_EXS),
            model='repeat_label',
            datatype='valid',
            num_examples=-1,
            display_examples=False,
            numthreads=2,
        )

        # old_out = sys.stdout
        # output = display_output()
        try:
            # sys.stdout = output
            report = eval_model(parser, printargs=False)
        finally:
            pass
            # restore sys.stdout
            # sys.stdout = old_out

        # str_output = str(output)
        print(report)
        self.assertEqual(report['total'], NUM_EXS)

if __name__ == '__main__':
    unittest.main()
