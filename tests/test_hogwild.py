# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.scripts.train_model import TrainLoop, run_eval, setup_args
from parlai.scripts.eval_model import eval_model

import ast
import unittest
import sys


class display_output(object):
    def __init__(self):
        self.data = []

    def write(self, s):
        self.data.append(s)

    def flush(self):
        pass

    def __str__(self):
        return "".join(self.data)


class TestHogwild(unittest.TestCase):
    """Check that hogwild is doing the right number of examples."""

    def test_hogwild_train(self):
        """Test the trainer eval with numthreads > 1 and batchsize in [1,2,3]."""
        parser = setup_args()
        NUM_EXS = 500
        parser.set_defaults(
            task='tasks.repeat:RepeatTeacher:{}'.format(1),
            evaltask='tasks.repeat:RepeatTeacher:{}'.format(NUM_EXS),
            model='repeat_label',
            num_examples=-1,
            display_examples=False,
            num_epochs=10,
        )

        old_out = sys.stdout
        output = display_output()
        try:
            sys.stdout = output
            for nt in [2, 5, 10]:
                parser.set_defaults(numthreads=nt)
                for bs in [1, 2, 3]:
                    parser.set_defaults(batchsize=bs)
                    parser.set_defaults(batch_sort=(bs % 2 == 0))
                    tl = TrainLoop(parser)
                    report_valid, report_test = tl.train()
                    # test final valid and test evals
                    self.assertEqual(report_valid['exs'], NUM_EXS)
                    self.assertEqual(report_test['exs'], NUM_EXS)

                    report_full, _world = run_eval(tl.agent, tl.opt, 'valid',
                        max_exs=-1, valid_world=tl.valid_world)
                    self.assertEqual(report_full['exs'], NUM_EXS)
                    report_part, _world = run_eval(tl.agent, tl.opt, 'valid',
                        max_exs=NUM_EXS / 5, valid_world=tl.valid_world)
                    self.assertTrue(report_part['exs'] < NUM_EXS)
        finally:
            # restore sys.stdout
            sys.stdout = old_out

    def test_hogwild_eval(self):
        """Test eval with numthreads > 1 and batchsize in [1,2,3]."""
        parser = setup_args()
        NUM_EXS = 500
        parser.set_defaults(
            task='tasks.repeat:RepeatTeacher:{}'.format(NUM_EXS),
            model='repeat_label',
            datatype='valid',
            num_examples=-1,
            display_examples=False,
        )

        old_out = sys.stdout
        output = display_output()
        try:
            sys.stdout = output
            for nt in [2, 5, 10]:
                parser.set_defaults(numthreads=nt)
                for bs in [1, 2, 3]:
                    parser.set_defaults(batchsize=bs)
                    parser.set_defaults(batch_sort=(bs % 2 == 0))
                    report = eval_model(parser, printargs=False)
                    self.assertEqual(report['exs'], NUM_EXS)
        finally:
            # restore sys.stdout
            sys.stdout = old_out

if __name__ == '__main__':
    unittest.main()
