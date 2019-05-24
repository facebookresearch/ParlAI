#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.thread_utils import SharedTable
from multiprocessing import Process
import unittest
import random
import time


class TestSharedTable(unittest.TestCase):
    """Make sure the package is alive."""

    def test_init_from_dict(self):
        d = {
            'a': 0,
            'b': 1,
            'c': 1.0,
            'd': True,
            1: False,
            2: 2.0
        }
        st = SharedTable(d)
        for k, v in d.items():
            assert(st[k] == v)

    def test_get_set_del(self):
        st = SharedTable({'key': 0})
        try:
            st['none']
            self.fail('did not fail on nonexistent key')
        except KeyError:
            pass

        st['key'] = 1
        assert st['key'] == 1

        st['key'] += 1
        assert st['key'] == 2

        try:
            st['key'] = 2.1
            self.fail('cannot change type of value for set keys')
        except TypeError:
            pass

        del st['key']
        assert 'key' not in st, 'key should have been removed from table'

        try:
            st['key'] = True
            self.fail('cannot change removed key')
        except KeyError:
            pass

    def test_iter_keys(self):
        st = SharedTable({'key': 0, 'ctr': 0.0, 'val': False, 'other': 1})
        assert len(st) == 4
        del st['key']
        assert len(st) == 3, 'length should decrease after deleting key'
        keyset1 = set(iter(st))
        keyset2 = set(st.keys())
        assert keyset1 == keyset2, 'iterating should return keys'
        assert len(keyset1) == 3, ''

    def test_concurrent_access(self):
        st = SharedTable({'cnt': 0})

        def inc():
            for _ in range(50):
                with st.get_lock():
                    st['cnt'] += 1
                time.sleep(random.randint(1, 5) / 10000)

        threads = []
        for _ in range(5):  # numthreads
            threads.append(Process(target=inc))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert st['cnt'] == 250

    def test_torch(self):
        try:
            import torch
        except ImportError:
            # pass by default if no torch available
            return

        st = SharedTable({'a': torch.FloatTensor([1]), 'b': torch.LongTensor(2)})
        assert st['a'][0] == 1.0
        assert len(st) == 2
        assert 'b' in st
        del st['b']
        assert 'b' not in st
        assert len(st) == 1

        if torch.cuda.is_available():
            st = SharedTable({
                'a': torch.cuda.FloatTensor([1]),
                'b': torch.cuda.LongTensor(2),
            })
            assert st['a'][0] == 1.0
            assert len(st) == 2
            assert 'b' in st
            del st['b']
            assert 'b' not in st
            assert len(st) == 1


if __name__ == '__main__':
    unittest.main()
