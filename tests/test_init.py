# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os
import unittest


class TestInit(unittest.TestCase):
    """Make sure the package is alive."""

    def test_init_everywhere(self):
        from parlai.core.params import ParlaiParser
        opt = ParlaiParser().parse_args()
        for root, subfolder, files in os.walk(os.path.join(opt['parlai_home'], 'parlai')):
            if not root.endswith('__pycache__'):
                assert '__init__.py' in files, 'Dir {} is missing __init__.py'.format(root)


if __name__ == '__main__':
    unittest.main()
