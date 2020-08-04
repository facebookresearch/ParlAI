#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import unittest

import parlai.utils.testing as testing_utils
from parlai.core.opt import Opt
from parlai.scripts.compare_opts import compare_opts

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

    def test_compare_opts(self):
        with testing_utils.tempdir() as tmpdir:

            # Define test opts
            opt1 = {
                'key0': (1, 2),
                'key1': 0,
                'key2': 'a',
                'key4': {'inner_key0': [1], 'inner_key1': True, 'inner_key2': 'yes'},
            }
            opt2 = {
                'key0': (1, 2),
                'key1': 1,
                'key3': 'b',
                'key4': {'inner_key0': [1], 'inner_key1': False, 'inner_key3': 'no'},
            }

            # Write test opts
            opt_dir = tmpdir
            opt_path_1 = os.path.join(opt_dir, '1.opt')
            opt_path_2 = os.path.join(opt_dir, '2.opt')
            with open(opt_path_1, 'w') as f1:
                json.dump(opt1, f1)
            with open(opt_path_2, 'w') as f2:
                json.dump(opt2, f2)

            # Compare opts
            output = compare_opts(opt_path_1=opt_path_1, opt_path_2=opt_path_2)
            desired_output = """
Args only found in opt 1:
key2: a

Args only found in opt 2:
key3: b

Args that are different in both opts:
key1:
    In opt 1: 0
    In opt 2: 1
key4 (printing only non-matching values in each dict):
    inner_key1:
        In opt 1: True
        In opt 2: False
    inner_key2:
        In opt 1: yes
        In opt 2: <MISSING>
    inner_key3:
        In opt 1: <MISSING>
        In opt 2: no
"""
            self.assertEqual(output, desired_output)
