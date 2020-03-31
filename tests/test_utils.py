#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.utils.misc import Timer, round_sigfigs, set_namedtuple_defaults
import parlai.utils.strings as string_utils
from copy import deepcopy
import time
import unittest


class TestUtils(unittest.TestCase):
    def test_round_sigfigs(self):
        x = 0
        y = 0
        assert round_sigfigs(x, 2) == y

        x = 100
        y = 100
        assert round_sigfigs(x, 2) == y

        x = 0.01
        y = 0.01
        assert round_sigfigs(x, 2) == y

        x = 0.00123
        y = 0.001
        assert round_sigfigs(x, 1) == y

        x = 0.37
        y = 0.4
        assert round_sigfigs(x, 1) == y

        x = 2353
        y = 2350
        assert round_sigfigs(x, 3) == y

        x = 3547345734
        y = 3547350000
        assert round_sigfigs(x, 6) == y

        x = 0.0000046246
        y = 0.00000462
        assert round_sigfigs(x, 3) == y

    def test_timer(self):
        t = Timer()
        time.sleep(1e-6)
        elapsed = t.stop().time()
        assert elapsed > 0

        same = t.time()
        assert elapsed == same

        t.resume()
        time.sleep(1e-6)
        more = t.time()
        assert more > elapsed

        rabbit = Timer()
        time.sleep(1e-6)
        turtle = Timer()
        time.sleep(1e-6)
        assert turtle.time() > 0
        assert turtle.time() < rabbit.time()

    def test_setnamedtupledefaults(self):
        from collections import namedtuple

        NT = namedtuple("NT", ("a", "b", "c"))

        # Shouldn't be able to construct a namedtuple without providing info
        try:
            NT()
            self.fail("Shouldn't be able to construct namedtuple")
        except TypeError:
            pass

        # Test setting default value
        set_namedtuple_defaults(NT)
        nt = NT()
        assert nt.a is None
        assert nt.b is None
        assert nt.c is None

        # Test setting it with something else
        set_namedtuple_defaults(NT, default=1)
        nt = NT()
        assert nt.a == 1
        assert nt.b == 1
        assert nt.c == 1

    def test_opt(self):
        opt = {'x': 0}
        opt = Opt(opt)
        opt['x'] += 1
        opt['x'] = 10
        self.assertEqual(opt.history[0][0], 'x', 'History not set properly')
        self.assertEqual(opt.history[0][1], 1, 'History not set properly')
        self.assertEqual(opt.history[1][0], 'x', 'History not set properly')
        self.assertEqual(opt.history[1][1], 10, 'History not set properly')

        opt_copy = deepcopy(opt)
        self.assertEqual(opt_copy.history[0][1], 1, 'Deepcopy history not set properly')
        self.assertEqual(
            opt_copy.history[1][1], 10, 'Deepcopy history not set properly'
        )


class TestStrings(unittest.TestCase):
    def test_normalize_reply_version1(self):
        assert string_utils.normalize_reply("I ' ve a cat .") == "I've a cat."
        assert (
            string_utils.normalize_reply("do you think i can dance?")
            == "Do you think I can dance?"
        )
        assert string_utils.normalize_reply("I ' m silly '") == "I'm silly'"

    def test_normalize_reply_version2(self):
        assert string_utils.normalize_reply("Add a period", 2) == "Add a period."
        assert string_utils.normalize_reply("Add a period?", 2) == "Add a period?"
        assert string_utils.normalize_reply("Add a period!", 2) == "Add a period!"
        assert string_utils.normalize_reply('"Add a period"', 2) == '"add a period"'

    def test_uppercase(self):
        assert string_utils.uppercase("this is a test") == "This is a test"
        assert string_utils.uppercase("tEst") == "TEst"


if __name__ == '__main__':
    unittest.main()
