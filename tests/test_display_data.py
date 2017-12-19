# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from examples.display_data import display_data
from parlai.core.params import ParlaiParser

import sys
import unittest


class TestDisplayData(unittest.TestCase):
    """Basic tests on the display_data.py example."""

    args = [
        '--task', 'babi:task1k:1',
    ]
    parser = ParlaiParser()
    opt = parser.parse_args(args, print_args=False)
    opt['num_examples'] = 1

    def test_output(self):
        """Does display_data reach the end of the loop?"""

        class display_output(object):
            def __init__(self):
                self.data = []

            def write(self, s):
                self.data.append(s)

            def __str__(self):
                return "".join(self.data)

        old_out = sys.stdout
        output = display_output()
        try:
            sys.stdout = output
            display_data(self.opt)
        finally:
            # restore sys.stdout
            sys.stdout = old_out

        str_output = str(output)
        self.assertTrue(len(str_output) > 0, "Output is empty")
        self.assertTrue("[babi:task1k:1]:" in str_output,
                        "Babi task did not print")
        self.assertTrue("~~" in str_output, "Example output did not complete")

if __name__ == '__main__':
    unittest.main()
