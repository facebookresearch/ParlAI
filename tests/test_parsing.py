#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from examples.eval_model import setup_args
import parlai.utils.testing as testing_utils
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
        with testing_utils.capture_output():
            parser.parse_args()
        help_str = parser.format_help()

        variant_start = help_str.find("Chooses locations of layer norms, etc")
        variant_end = help_str[variant_start:].find("\n")
        variant_str = help_str[variant_start: variant_start + variant_end].split()
        assert(variant_str[variant_str.index("(recommended:") + 1][:-1] == "xlm")


if __name__ == '__main__':
    unittest.main()
