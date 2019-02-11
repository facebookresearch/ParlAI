#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from examples.display_data import display_data
from parlai.core.params import ParlaiParser
import unittest
import parlai.core.testing_utils as testing_utils


class TestDisplayData(unittest.TestCase):
    """Basic tests on the display_data.py example."""
    def test_output(self):
        """Does display_data reach the end of the loop?"""
        with testing_utils.capture_output() as stdout:
            parser = ParlaiParser()
            opt = parser.parse_args(['--task', 'babi:task1k:1'], print_args=False)
            opt['num_examples'] = 1
            display_data(opt)

        str_output = stdout.getvalue()
        self.assertGreater(len(str_output), 0, "Output is empty")
        self.assertIn("[babi:task1k:1]:", str_output, "Babi task did not print")
        self.assertIn("~~", str_output, "Example output did not complete")


if __name__ == '__main__':
    unittest.main()
