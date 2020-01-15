#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test TorchRankerAgent.
"""

import os
import unittest

import parlai.utils.testing as testing_utils
from parlai.tasks.integration_tests.agents import CandidateTeacher


class _AbstractTRATest(unittest.TestCase):
    """
    Test upgrade_opt behavior.
    """

    @classmethod
    def setUpClass(cls):
        if cls is _AbstractTRATest:
            raise unittest.SkipTest('Skip abstract parent class')
        super(_AbstractTRATest, cls).setUpClass()

    def _get_args(self):
        # Add arguments for the Torch Ranker Agent to test
        # Override in child classes
        return dict(
            task='integration_tests:candidate',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=16,
            embedding_size=32,
            num_epochs=4,
        )

    def _get_threshold(self):
        # Accuracy threshold
        return 0.8

    # test train inline cands
    @testing_utils.retry(ntries=3)
    def test_train_inline(self):
        args = self._get_args()
        args['candidates'] = 'inline'
        valid, test = testing_utils.train_model(args)
        threshold = self._get_threshold()

        self.assertGreaterEqual(valid['hits@1'], threshold)

    # test train batch cands
    @testing_utils.retry(ntries=3)
    def test_train_batch(self):
        args = self._get_args()
        args['candidates'] = 'batch'
        valid, test = testing_utils.train_model(args)
        threshold = self._get_threshold()

        self.assertGreaterEqual(valid['hits@1'], threshold)

    # test train fixed
    @testing_utils.retry(ntries=3)
    def test_train_fixed(self):
        args = self._get_args()
        args['candidates'] = 'fixed'
        args['encode_candidate_vecs'] = False
        valid, test = testing_utils.train_model(args)
        threshold = self._get_threshold()

        self.assertGreaterEqual(valid['hits@1'], threshold)

    # test train batch all cands
    @testing_utils.retry(ntries=3)
    def test_train_batch_all(self):
        args = self._get_args()
        args['candidates'] = 'batch-all-cands'
        valid, test = testing_utils.train_model(args)
        threshold = self._get_threshold()

        self.assertGreaterEqual(valid['hits@1'], threshold)

    # test eval inline ecands
    @testing_utils.retry(ntries=3)
    def test_eval_inline(self):
        args = self._get_args()
        args['eval_candidates'] = 'inline'
        valid, test = testing_utils.train_model(args)
        threshold = self._get_threshold()

        self.assertGreaterEqual(valid['hits@1'], threshold)

    # test eval batch ecands
    @testing_utils.retry(ntries=3)
    def test_eval_batch(self):
        args = self._get_args()
        args['eval_candidates'] = 'batch'
        valid, test = testing_utils.train_model(args)
        threshold = self._get_threshold()

        self.assertGreaterEqual(valid['hits@1'], threshold)

    # test eval fixed ecands
    @testing_utils.retry(ntries=3)
    def test_eval_fixed(self):
        args = self._get_args()
        args['eval_candidates'] = 'fixed'
        args['encode_candidate_vecs'] = True
        args['ignore_bad_candidates'] = True
        valid, test = testing_utils.train_model(args)

        # none of the train candidates appear in evaluation, so should have
        # zero accuracy: this tests whether the fixed candidates were built
        # properly (i.e., only using candidates from the train set)
        self.assertEqual(valid['hits@1'], 0)

        # now try again with a fixed candidate file that includes all possible
        # candidates
        teacher = CandidateTeacher({'datatype': 'train'})
        all_cands = teacher.train + teacher.val + teacher.test
        all_cands_str = '\n'.join([' '.join(x) for x in all_cands])

        with testing_utils.tempdir() as tmpdir:
            tmp_cands_file = os.path.join(tmpdir, 'all_cands.text')
            with open(tmp_cands_file, 'w') as f:
                f.write(all_cands_str)
            args['fixed_candidates_path'] = tmp_cands_file
            args['encode_candidate_vecs'] = False  # don't encode before training
            args['ignore_bad_candidates'] = False
            args['num_epochs'] = 4
            valid, test = testing_utils.train_model(args)
            self.assertGreaterEqual(valid['hits@100'], 0.1)

    # test eval vocab ecands
    @testing_utils.retry(ntries=3)
    def test_eval_vocab(self):
        args = self._get_args()
        args['eval_candidates'] = 'vocab'
        args['encode_candidate_vecs'] = True
        valid, test = testing_utils.train_model(args)

        # accuracy should be zero, none of the vocab candidates should be the
        # correct label
        self.assertEqual(valid['hits@100'], 0)


class TestTransformerRanker(_AbstractTRATest):
    def _get_args(self):
        args = super()._get_args()
        new_args = dict(
            model='transformer/ranker',
            n_layers=1,
            n_heads=4,
            ffn_size=64,
            gradient_clip=0.5,
        )
        for k, v in new_args.items():
            args[k] = v
        return args


class TestMemNN(_AbstractTRATest):
    def _get_args(self):
        args = super()._get_args()
        args['model'] = 'memnn'
        return args

    def _get_threshold(self):
        # this is a slightly worse model, so we expect it to perform worse
        return 0.5


class TestPolyRanker(_AbstractTRATest):
    def _get_args(self):
        args = super()._get_args()
        new_args = dict(
            model='transformer/polyencoder',
            n_layers=1,
            n_heads=4,
            ffn_size=64,
            gradient_clip=0.5,
        )
        for k, v in new_args.items():
            args[k] = v
        return args

    def _get_threshold(self):
        return 0.6

    def test_eval_fixed_label_not_in_cands(self):
        # test where cands during eval do not contain test label
        args = self._get_args()
        args[
            'model'
        ] = 'parlai.agents.transformer.polyencoder:IRFriendlyPolyencoderAgent'
        args['eval_candidates'] = 'fixed'

        teacher = CandidateTeacher({'datatype': 'train'})
        all_cands = teacher.train + teacher.val + teacher.test
        train_val_cands = teacher.train + teacher.val
        all_cands_str = '\n'.join([' '.join(x) for x in all_cands])
        train_val_cands_str = '\n'.join([' '.join(x) for x in train_val_cands])

        with testing_utils.tempdir() as tmpdir:
            tmp_cands_file = os.path.join(tmpdir, 'all_cands.text')
            with open(tmp_cands_file, 'w') as f:
                f.write(all_cands_str)
            tmp_train_val_cands_file = os.path.join(tmpdir, 'train_val_cands.text')
            with open(tmp_train_val_cands_file, 'w') as f:
                f.write(train_val_cands_str)
            args['fixed_candidates_path'] = tmp_cands_file
            args['encode_candidate_vecs'] = False  # don't encode before training
            args['ignore_bad_candidates'] = False
            args['model_file'] = os.path.join(tmpdir, 'model')
            args['dict_file'] = os.path.join(tmpdir, 'model.dict')
            args['num_epochs'] = 4
            # Train model where it has access to the candidate in labels
            valid, test = testing_utils.train_model(args)
            self.assertGreaterEqual(valid['hits@100'], 0.0)

            # Evaluate model where label is not in fixed candidates
            args['fixed_candidates_path'] = tmp_train_val_cands_file

            # Will fail without appropriate arg set
            with self.assertRaises(RuntimeError):
                testing_utils.eval_model(args, skip_valid=True)

            args['add_label_to_fixed_cands'] = True
            valid, test = testing_utils.eval_model(args, skip_valid=True)
            self.assertGreaterEqual(test['hits@100'], 0.0)


if __name__ == '__main__':
    unittest.main()
