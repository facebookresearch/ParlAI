#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.utils import Timer
from parlai.core.utils import round_sigfigs
from parlai.core.utils import set_namedtuple_defaults
from parlai.core.utils import padded_tensor
from parlai.core.utils import argsort
import time
import unittest
import torch
import numpy as np


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

    def test_padded_tensor(self):
        # list of lists
        lol = [[1, 2], [3, 4, 5]]
        output, lens = padded_tensor(lol)
        assert np.all(output.numpy() == np.array([[1, 2, 0], [3, 4, 5]]))
        assert lens == [2, 3]
        output, _ = padded_tensor(lol, left_padded=True)
        assert np.all(output.numpy() == np.array([[0, 1, 2], [3, 4, 5]]))
        output, _ = padded_tensor(lol, pad_idx=99)
        assert np.all(output.numpy() == np.array([[1, 2, 99], [3, 4, 5]]))

    def test_argsort(self):
        keys = [5, 4, 3, 2, 1]
        items = ["five", "four", "three", "two", "one"]
        items2 = ["e", "d", "c", "b", "a"]
        torch_keys = torch.LongTensor(keys)
        assert argsort(keys, items, items2) == [
            list(reversed(items)), list(reversed(items2))
        ]
        assert argsort(keys, items, items2, descending=True) == [items, items2]

        assert np.all(argsort(torch_keys, torch_keys)[0].numpy() == np.arange(1, 6))


if __name__ == '__main__':
    unittest.main()
