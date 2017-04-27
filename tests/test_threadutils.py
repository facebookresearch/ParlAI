# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
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
            'd': 'hello',
            1: 'world',
            2: 2.0
        }
        st = SharedTable(d)
        for k, v in d.items():
            assert(st[k] == v)

    def test_get_set_del(self):
        st = SharedTable()
        try:
            st['key']
            assert False, 'did not fail on nonexistent key'
        except KeyError:
            pass

        st['key'] = 1
        assert st['key'] == 1

        st['key'] += 1
        assert st['key'] == 2

        try:
            st['key'] = 2.1
            assert False, 'cannot change type of value for set keys'
        except TypeError:
            pass

        del st['key']
        assert 'key' not in st, 'key should have been removed from table'

        st['key'] = 'hello'
        assert st['key'] == 'hello'

        st['key'] += ' world'
        assert st['key'] == 'hello world'

        st['ctr'] = 0
        keyset1 = set(iter(st))
        keyset2 = set(st.keys())
        assert keyset1 == keyset2, 'iterating should return keys'

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


if __name__ == '__main__':
    unittest.main()
