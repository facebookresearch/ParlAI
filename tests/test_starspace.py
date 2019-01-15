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

NUM_EPOCHS = 3
NUM_THREADS = 10


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


class TestStarspace(unittest.TestCase):
    """Checks that seq2seq can learn some very basic tasks."""

    def test_labelcands_starspace(self):
        """This test uses a single-turn task, so doesn't test memories."""

        stdout, valid, test = _mock_train(
            task='integration_tests:CandidateTeacher',
            model='starspace',
            num_epochs=NUM_EPOCHS,
            numthreads=NUM_THREADS,
            no_cuda=True,
            embedding_size=32,
        )

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    def test_labelcands_multi_starspace(self):
        """This test uses a multi-turn task and multithreading."""
        stdout, valid, test = _mock_train(
            task='integration_tests:MultiturnCandidateTeacher',
            model='starspace',
            num_epochs=NUM_THREADS,
            numthreads=10,
            no_cuda=True,
            embedding_size=32,
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
