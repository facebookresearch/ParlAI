#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.scripts.train_model import TrainLoop, run_eval, setup_args, load_eval_world
from parlai.scripts.eval_model import eval_model

import unittest
import sys

import torch


SKIP_HOGWILD = torch.cuda.device_count() > 0


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

    @unittest.skipIf(SKIP_HOGWILD, "No hogwild tests if GPUs are available.")
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
                    tl = TrainLoop(parser)
                    report_valid, report_test = tl.train()
                    # test final valid and test evals
                    self.assertEqual(report_valid['exs'], NUM_EXS)
                    self.assertEqual(report_test['exs'], NUM_EXS)

                    valid_world = load_eval_world(tl.agent, tl.opt, 'valid')
                    report_full = run_eval(valid_world, tl.opt, 'valid', max_exs=-1)
                    self.assertEqual(report_full['exs'], NUM_EXS)
                    valid_world = load_eval_world(tl.agent, tl.opt, 'valid')
                    report_part = run_eval(
                        valid_world, tl.opt, 'valid',
                        max_exs=NUM_EXS / 5
                    )
                    self.assertTrue(report_part['exs'] < NUM_EXS)
        finally:
            # restore sys.stdout
            sys.stdout = old_out

    @unittest.skipIf(SKIP_HOGWILD, "No hogwild tests if GPUs are available.")
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
                    report = eval_model(parser, printargs=False)
                    self.assertEqual(report['exs'], NUM_EXS)
        finally:
            # restore sys.stdout
            sys.stdout = old_out


if __name__ == '__main__':
    unittest.main()
