#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import parlai.utils.testing as testing_utils
from parlai.scripts.build_candidates import BuildCandidates


class TestStarspace(unittest.TestCase):
    def test_training_bs1(self):
        valid, test = testing_utils.train_model(
            {'model': 'starspace', 'task': 'integration_tests', 'num_epochs': 1.0}
        )

        assert valid['hits@1'] > 0.5
        assert test['hits@1'] > 0.5

    def test_training_otheropt(self):
        valid, test = testing_utils.train_model(
            {
                'model': 'starspace',
                'task': 'integration_tests',
                'num_epochs': 1.0,
                'input_dropout': 0.1,
                'share_embeddings': False,
                'lins': 1,
                'tfidf': True,
            }
        )

        assert valid['hits@1'] > 0.25
        assert test['hits@1'] > 0.25

    def test_training_fixedcand(self):
        with testing_utils.tempdir() as tmpdir:
            valid, test = testing_utils.train_model(
                {
                    'model': 'starspace',
                    'task': 'integration_tests',
                    'num_epochs': 1.0,
                    'dict_file': os.path.join(tmpdir, 'model.dict'),
                    'model_file': os.path.join(tmpdir, 'model'),
                }
            )

            assert valid['accuracy'] > 0.25
            assert test['accuracy'] > 0.25

            cand = os.path.join(tmpdir, 'cands.txt')
            BuildCandidates.main(task='integration_tests', outfile=cand)
            valid, test = testing_utils.eval_model(
                {
                    'model': 'starspace',
                    'task': 'integration_tests:nocandidate',
                    'model_file': os.path.join(tmpdir, 'model'),
                    'dict_file': os.path.join(tmpdir, 'model.dict'),
                    'fixed_candidates_file': os.path.join(tmpdir, 'cands.txt'),
                }
            )
            assert valid['f1'] >= 0.5
            assert test['f1'] >= 0.5
            assert valid['accuracy'] == 0.0
            assert test['accuracy'] == 0.0
