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

BATCH_SIZE = 1
NUM_EPOCHS = 3
LR = 1


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


class TestMemnn(unittest.TestCase):
    """Checks that seq2seq can learn some very basic tasks."""

    def test_labelcands_nomemnn(self):
        """This test uses a single-turn task, so doesn't test memories."""

        stdout, valid, test = _mock_train(
            task='integration_tests:CandidateTeacher',
            model='memnn',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            numthreads=1,
            no_cuda=True,
            embedding_size=32,
            gradient_clip=1.0,
            hops=1,
            position_encoding=True,
            use_time_features=False,
            memsize=0,
            rank_candidates=True,
        )

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    def test_labelcands_multi(self):
        """This test uses a multi-turn task and multithreading."""
        stdout, valid, test = _mock_train(
            task='integration_tests:MultiturnCandidateTeacher',
            model='memnn',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS * 3,
            numthreads=4,
            no_cuda=True,
            embedding_size=32,
            gradient_clip=1.0,
            hops=2,
            position_encoding=False,
            use_time_features=True,
            memsize=5,
            rank_candidates=True,
        )

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )


if __name__ == '__main__':
    unittest.main()
