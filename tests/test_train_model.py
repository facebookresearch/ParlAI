#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic tests that ensure train_model.py behaves in predictable ways.
"""

import unittest
import parlai.core.testing_utils as testing_utils


class TestTrainModel(unittest.TestCase):
    def test_fast_final_eval(self):
        stdout, valid, test = testing_utils.train_model({
            'task': 'integration_tests',
            'validation_max_exs': 10,
            'model': 'repeat_label',
            'short_final_eval': True,
            'num_epochs': 1.0,
        })
        self.assertEqual(valid['exs'], 10, 'Validation exs is wrong')
        self.assertEqual(test['exs'], 10, 'Test exs is wrong')


if __name__ == '__main__':
    unittest.main()
