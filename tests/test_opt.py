#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import parlai.utils.testing as testing_utils
from parlai.core.opt import Opt

"""
Test Opt and related mechanisms.
"""


class TestOpt(unittest.TestCase):
    def test_save_load(self):
        o = Opt({'a': 3, 'b': 'foo'})
        with testing_utils.tempdir() as tmpdir:
            fn = os.path.join(tmpdir, "opt")
            o.save(fn)
            o2 = Opt.load(fn)
            assert o == o2

    def test_save_withignore(self):
        o = Opt({'a': 3, 'b': 'foo', 'override': {'a': 3}})
        with testing_utils.tempdir() as tmpdir:
            fn = os.path.join(tmpdir, "opt")
            o.save(fn)
            o2 = Opt.load(fn)
            assert o != o2
            assert 'override' not in o2
