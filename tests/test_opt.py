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

    # Define test opts for opt comparison script
    compare_opt_1 = {
        'key0': (1, 2),
        'key1': 0,
        'key2': 'a',
        'override': {'inner_key0': [1], 'inner_key1': True, 'inner_key2': 'yes'},
    }
    compare_opt_2 = {
        'key0': (1, 2),
        'key1': 1,
        'key3': 'b',
        'override': {'inner_key0': [1], 'inner_key1': False, 'inner_key3': 'no'},
    }

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
        """
        Compare opts by loading them with Opt.load().

        Will not compare the override field.
        """

        with testing_utils.tempdir() as tmpdir:
            # Write test opts
            opt_dir = tmpdir
            opt_path_1 = os.path.join(opt_dir, '1.opt')
            opt_path_2 = os.path.join(opt_dir, '2.opt')
            with open(opt_path_1, 'w') as f1:
                json.dump(self.compare_opt_1, f1)
            with open(opt_path_2, 'w') as f2:
                json.dump(self.compare_opt_2, f2)

            # Compare opts
            output = compare_opts(opt_path_1=opt_path_1, opt_path_2=opt_path_2)
            desired_output = """
Args only found in opt 1:
key2: a

Args only found in opt 2:
key3: b

Args that are different in both opts:
key1:
\tIn opt 1: 0
\tIn opt 2: 1"""
            self.assertEqual(output, desired_output)

    def test_compare_opts_load_raw(self):
        """
        Compare opts by loading them from JSON instead of with Opt.load().

        Will compare the override field.
        """

        with testing_utils.tempdir() as tmpdir:

            # Write test opts
            opt_dir = tmpdir
            opt_path_1 = os.path.join(opt_dir, '1.opt')
            opt_path_2 = os.path.join(opt_dir, '2.opt')
            with open(opt_path_1, 'w') as f1:
                json.dump(self.compare_opt_1, f1)
            with open(opt_path_2, 'w') as f2:
                json.dump(self.compare_opt_2, f2)

            # Compare opts
            output = compare_opts(
                opt_path_1=opt_path_1, opt_path_2=opt_path_2, load_raw=True
            )
            desired_output = """
Args only found in opt 1:
key2: a

Args only found in opt 2:
key3: b

Args that are different in both opts:
key1:
\tIn opt 1: 0
\tIn opt 2: 1
override (printing only non-matching values in each dict):
\tinner_key1:
\t\tIn opt 1: True
\t\tIn opt 2: False
\tinner_key2:
\t\tIn opt 1: yes
\t\tIn opt 2: <MISSING>
\tinner_key3:
\t\tIn opt 1: <MISSING>
\t\tIn opt 2: no"""
            self.assertEqual(output, desired_output)
