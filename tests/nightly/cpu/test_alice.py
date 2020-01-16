#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import unittest
import parlai.utils.testing as testing_utils


class TestAlice(unittest.TestCase):
    def test_alice_runs(self):
        """
        Test that the ALICE agent is stable over time.
        """
        valid, test = testing_utils.eval_model(dict(task='convai2', model='alice'))
        self.assertEqual(valid['f1'], 0.01397)


if __name__ == '__main__':
    unittest.main()
