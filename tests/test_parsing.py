#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from examples.eval_model import setup_args
import unittest


class TestRecommended(unittest.TestCase):
    """Basic tests on the eval_model.py example."""

    def test_output(self):
        """Test output of running eval_model"""
        parser = setup_args()
        parser.set_defaults(
            task='integration_tests',
            model='transformer/polyencoder',
            datatype='valid',
            num_examples=5,
            display_examples=False,
        )
        parser.format_help()


if __name__ == '__main__':
    unittest.main()
