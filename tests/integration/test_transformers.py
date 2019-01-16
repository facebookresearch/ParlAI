#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import io
import contextlib
import tempfile
import os
import shutil

from parlai.scripts.train_model import TrainLoop, setup_args


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


class TestTransformerRanker(unittest.TestCase):
    """Checks that transformer_ranker can learn some very basic tasks."""

    def test_repeater(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:CandidateTeacher',
            model='transformer/ranker',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=32,
            num_epochs=10,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            scores_norm='dot',
        )

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )


class TestTransformerGenerator(unittest.TestCase):
    """Checks that the generative transformer can learn basic tasks."""

    def test_ranking(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:CandidateTeacher',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=32,
            num_epochs=10,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
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

    def test_generation(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:NocandidateTeacher',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=32,
            num_epochs=15,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            beam_size=5,
        )

        self.assertTrue(
            valid['ppl'] < 1.20,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            valid['bleu'] >= .95,
            "valid blue = {}\nLOG:\n{}".format(valid['bleu'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 1.20,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )
        self.assertTrue(
            test['bleu'] >= .95,
            "test bleu = {}\nLOG:\n{}".format(test['bleu'], stdout)
        )


if __name__ == '__main__':
    unittest.main()
