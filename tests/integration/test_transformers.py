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


def _mock_train(outdir=None, keepoutdir=False, override=None, **args):
    if not outdir:
        outdir = tempfile.mkdtemp()
    parser = setup_args()
    parser.set_defaults(
        model_file=os.path.join(outdir, "model"),
        **args,
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        opt = parser.parse_args(print_args=False)
        if override:
            opt['override'] = override
        tl = TrainLoop(opt)
        valid, test = tl.train()
    if not keepoutdir:
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
            num_epochs=15,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            scores_norm='dot',
        )

        self.assertGreaterEqual(
            valid['hits@1'],
            0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertGreaterEqual(
            test['hits@1'],
            0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    def test_resuming(self):
        outdir = tempfile.mkdtemp()
        stdout1, valid1, test1 = _mock_train(
            outdir=outdir,
            keepoutdir=True,
            task='integration_tests:CandidateTeacher',
            model='transformer/ranker',
            optimizer='adamax',
            learningrate=1e-3,
            batchsize=32,
            num_epochs=1,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            scores_norm='dot',
            warmup_updates=1,
            lr_scheduler='invsqrt',
        )
        stdout2, valid2, test2 = _mock_train(
            outdir=outdir,
            keepoutdir=False,
            task='integration_tests:CandidateTeacher',
            model='transformer/ranker',
            optimizer='adamax',
            learningrate=1e-3,
            batchsize=32,
            num_epochs=1,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            scores_norm='dot',
            warmup_updates=1,
            lr_scheduler='invsqrt',
        )
        # make sure the number of updates is being tracked correctly
        self.assertGreater(
            valid2['num_updates'],
            valid1['num_updates'],
            'Number of updates is not increasing'
        )
        # make sure the learning rate is decreasing
        self.assertLess(
            valid2['lr'],
            valid1['lr'],
            'Learning rate is not decreasing'
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

        self.assertGreaterEqual(
            valid['hits@1'],
            0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertGreaterEqual(
            test['hits@1'],
            0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    def test_generation(self):
        stdout, valid, test = _mock_train(
            task='integration_tests:NocandidateTeacher',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=32,
            num_epochs=20,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            beam_size=5,
        )

        self.assertLessEqual(
            valid['ppl'],
            1.20,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertGreaterEqual(
            valid['bleu'],
            0.95,
            "valid blue = {}\nLOG:\n{}".format(valid['bleu'], stdout)
        )
        self.assertLessEqual(
            test['ppl'],
            1.20,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )
        self.assertGreaterEqual(
            test['bleu'],
            0.95,
            "test bleu = {}\nLOG:\n{}".format(test['bleu'], stdout)
        )

    def test_resuming(self):
        BASE_ARGS = dict(
            task='integration_tests:NocandidateTeacher',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=1e-3,
            batchsize=32,
            num_epochs=1,
            no_cuda=True,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            skip_generation=True,
            lr_scheduler='invsqrt',
            warmup_updates=1,
        )

        outdir = tempfile.mkdtemp()
        stdout1, valid1, testr1 = _mock_train(
            outdir=outdir,
            keepoutdir=True,
            **BASE_ARGS,
        )
        stdout2, valid2, test2 = _mock_train(
            outdir=outdir,
            keepoutdir=True,
            **BASE_ARGS,
        )
        # make sure the number of updates is being tracked correctly
        self.assertGreater(
            valid2['num_updates'],
            valid1['num_updates'],
            'Number of updates is not increasing'
        )
        # make sure the learning rate is decreasing
        self.assertLess(
            valid2['lr'],
            valid1['lr'],
            'Learning rate is not decreasing'
        )
        # but make sure we're not loading the scheduler if we're fine tuning
        stdout3, valid3, test3 = _mock_train(
            init_model=os.path.join(outdir, 'model'),
            keepoutdir=False,
            **BASE_ARGS,
        )
        self.assertEqual(
            valid3['num_updates'],
            valid1['num_updates'],
            'Finetuning LR scheduler reset failed (num_updates).'
        )
        self.assertEqual(
            valid3['lr'],
            valid1['lr'],
            'Finetuning LR scheduler reset failed (lr).'
        )
        # and make sure we're not loading the scheduler if it changes
        stdout4, valid4, test4 = _mock_train(
            init_model=os.path.join(outdir, 'model'),
            outdir=outdir,
            keepoutdir=True,
            lr_scheduler='reduceonplateau',
            **BASE_ARGS,
        )
        self.assertEqual(
            valid4['num_updates'],
            valid1['num_updates'],
            'LR scheduler change reset failed (num_updates).\n' + stdout4
        )
        self.assertEqual(
            valid4['lr'],
            1e-3,
            'LR is not correct in final resume.\n' + stdout4
        )


if __name__ == '__main__':
    unittest.main()
