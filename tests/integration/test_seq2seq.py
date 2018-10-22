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

BATCH_SIZE = 16
NUM_EPOCHS = 10
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


class TestSeq2Seq(unittest.TestCase):
    """Checks that seq2seq can learn some very basic tasks."""

    def test_generation(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:NocandidateTeacher',
            model='seq2seq',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            numthreads=1,
            no_cuda=True,
            embeddingsize=16,
            hiddensize=16,
            rnn_class='gru',
            attention='general',
            gradient_clip=1.0,
            dropout=0.0,
            lookuptable='all',
        )

        self.assertTrue(
            valid['ppl'] < 1.2,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 1.2,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )

    def test_generation_multithreaded(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:NocandidateTeacher',
            model='seq2seq',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            numthreads=2,
            no_cuda=True,
            embeddingsize=16,
            hiddensize=16,
            rnn_class='gru',
            attention='general',
            gradient_clip=1.0,
            dropout=0.0,
            lookuptable='all',
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
