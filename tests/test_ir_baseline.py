#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import parlai.utils.testing as testing_utils
from parlai.scripts.build_candidates import BuildCandidates


class TestIrBaseline(unittest.TestCase):
    def test_integration(self):
        valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests',
                'model': 'ir_baseline',
                'batchsize': 1,
                'datatype': 'train:ordered',
                'num_epochs': 1,
            }
        )
        assert valid['f1'] >= 0.99
        assert test['f1'] >= 0.99

    def test_babi(self):
        valid, test = testing_utils.train_model(
            {
                'task': 'babi:task1k:1',
                'model': 'ir_baseline',
                'batchsize': 1,
                'datatype': 'train:ordered',
                'num_epochs': 1,
            }
        )
        assert valid['f1'] == 0.41
        assert test['f1'] >= 0.437

    def test_fixed_label(self):
        with testing_utils.tempdir() as tmpdir:
            testing_utils.train_model(
                {
                    'task': 'integration_tests',
                    'model': 'ir_baseline',
                    'batchsize': 1,
                    'datatype': 'train:ordered',
                    'num_epochs': 1,
                    'model_file': os.path.join(tmpdir, 'model'),
                }
            )
            with open(os.path.join(tmpdir, 'cands.txt'), 'w') as f:
                f.write("1 2 3 4\n")
                f.write("4 5 6 7\n")
            valid, test = testing_utils.eval_model(
                {
                    'task': 'integration_tests',
                    'model': 'ir_baseline',
                    'model_file': os.path.join(tmpdir, 'model'),
                    'label_candidates_file': os.path.join(tmpdir, 'cands.txt'),
                }
            )
            assert valid['f1'] == 0.6175
            assert test['f1'] == 0.625

    def test_fixed_label2(self):
        with testing_utils.tempdir() as tmpdir:
            testing_utils.train_model(
                {
                    'task': 'integration_tests',
                    'model': 'ir_baseline',
                    'batchsize': 1,
                    'datatype': 'train:ordered',
                    'num_epochs': 1,
                    'model_file': os.path.join(tmpdir, 'model'),
                }
            )
            cand = os.path.join(tmpdir, 'cands.txt')
            BuildCandidates.main(task='integration_tests', outfile=cand)
            valid, test = testing_utils.eval_model(
                {
                    'task': 'integration_tests',
                    'model': 'ir_baseline',
                    'model_file': os.path.join(tmpdir, 'model'),
                    'label_candidates_file': os.path.join(tmpdir, 'cands.txt'),
                }
            )
            assert valid['f1'] == 1.0
            assert test['f1'] == 1.0
            assert valid['accuracy'] == 0.0
            assert test['accuracy'] == 0.0
