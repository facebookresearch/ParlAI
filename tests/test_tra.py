#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test TorchRankerAgent."""

import os
import unittest

from parlai.core.agents import create_agent
import parlai.core.testing_utils as testing_utils
from parlai.core.params import ParlaiParser


ParlaiParser()  # instantiate to set PARLAI_HOME environment var
HOME_DIR = str(os.environ['PARLAI_HOME'])


def get_train_args():
    return dict(
        task='integration_tests:candidate',
        model='transformer/ranker',
        optimizer='adamax',
        learningrate=7e-3,
        batchsize=16,
        num_epochs=4,
        n_layers=1,
        n_heads=4,
        ffn_size=64,
        embedding_size=32,
        gradient_clip=0.5,
    )


class TestTorchRankerAgent(unittest.TestCase):
    """Test upgrade_opt behavior."""

    # test train inline cands
    def test_train_inline(self):
        args = get_train_args()
        args['candidates'] = 'inline'
        stdout, valid, test = testing_utils.train_model(args)

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

    # test train batch cands
    def test_train_batch(self):
        args = get_train_args()
        args['candidates'] = 'batch'
        stdout, valid, test = testing_utils.train_model(args)

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

    # test train fixed
    def test_train_fixed(self):
        args = get_train_args()
        args['candidates'] = 'fixed'
        args['encode_candidate_vecs'] = False
        stdout, valid, test = testing_utils.train_model(args)

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

    # test train batch all cands
    def test_train_batch_all(self):
        args = get_train_args()
        args['candidates'] = 'batch-all-cands'
        stdout, valid, test = testing_utils.train_model(args)

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

    # test eval inline ecands
    def test_eval_inline(self):
        args = get_train_args()
        args['eval_candidates'] = 'inline'
        stdout, valid, test = testing_utils.train_model(args)

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

    # test eval batch ecands
    def test_eval_batch(self):
        args = get_train_args()
        args['eval_candidates'] = 'batch'
        stdout, valid, test = testing_utils.train_model(args)

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

    # test eval fixed ecands
    def test_eval_fixed(self):
        args = get_train_args()
        args['eval_candidates'] = 'fixed'
        args['encode_candidate_vecs'] = True
        args['ignore_bad_candidates'] = True
        stdout, valid, test = testing_utils.train_model(args)

        # none of the train candidates appear in evaluation, so should have
        # zero accuracy: this tests whether the fixed candidates were built
        # properly (i.e., only using candidates from the train set)
        self.assertEqual(
            valid['hits@1'],
            0,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

        # now try again with a fixed candidate file that includes all possible
        # candidates
        args['fixed_candidates_path'] = os.path.join(
            HOME_DIR,
            'parlai/tasks/integration_tests/all_cands.txt'
        )
        args['ignore_bad_candidates'] = False
        args['num_epochs'] = 20
        stdout, valid, test = testing_utils.train_model(args)
        self.assertGreaterEqual(
            valid['hits@100'],
            0.4,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )

    # test eval vocab ecands
    def test_eval_vocab(self):
        args = get_train_args()
        args['eval_candidates'] = 'vocab'
        args['encode_candidate_vecs'] = True
        stdout, valid, test = testing_utils.train_model(args)

        # accuracy should be zero, none of the vocab candidates should be the
        # correct label
        self.assertEqual(
            valid['hits@100'],
            0,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )


if __name__ == '__main__':
    unittest.main()
