#!/usr/bin/env python3

import unittest
import parlai.utils.testing as testing_utils


class TestUnigram(unittest.TestCase):
    def test_unigram(self):
        valid, test = testing_utils.train_model(
            {'model': 'unigram', 'task': 'integration_tests', 'num_epochs': 0.01}
        )
        assert valid['f1'] > 0
