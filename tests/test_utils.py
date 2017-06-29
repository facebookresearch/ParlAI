# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.utils import Timer, round_sigfigs
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
        elapsed = t.stop().time()
        assert elapsed > 0

        same = t.time()
        assert elapsed == same

        t.resume()
        time.sleep(0.1)
        more = t.time()
        assert more > elapsed

        other = Timer()
        less = other.reset().time()
        assert less > 0
        assert less < t.time()


if __name__ == '__main__':
    unittest.main()
