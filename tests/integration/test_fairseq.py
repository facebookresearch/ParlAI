#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import unittest
import io
import contextlib
import tempfile
import os
import shutil

from parlai.scripts.train_model import TrainLoop, setup_args

SKIP_TESTS = False
try:
    import fairseq  # noqa: F401
except ImportError:
    SKIP_TESTS = True


BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-2


def _mock_train(**args):
    outdir = tempfile.mkdtemp()
    parser = setup_args()
    parser.set_defaults(
        model_file=os.path.join(outdir, "model"),
        **args,
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        tl = TrainLoop(parser.parse_args(print_args=False))
        valid, test = tl.train()

    shutil.rmtree(outdir)
    return stdout.getvalue(), valid, test


class TestFairseq(unittest.TestCase):
    """Checks that fairseq can learn some very basic tasks."""

    @unittest.skipIf(SKIP_TESTS, "Fairseq not installed")
    def test_labelcands(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:CandidateTeacher',
            model='fairseq',
            arch='lstm_wiseman_iwslt_de_en',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            rank_candidates=True,
            skip_generation=True,
        )

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    @unittest.skipIf(SKIP_TESTS, "Fairseq not installed")
    def test_generation(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:NocandidateTeacher',
            model='fairseq',
            arch='lstm_wiseman_iwslt_de_en',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            rank_candidates=False,
            skip_generation=False,
        )

        self.assertTrue(
            valid['ppl'] < 1.2,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 1.2,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )


if __name__ == '__main__':
    unittest.main()
