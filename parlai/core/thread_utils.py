# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Provides utilities useful for multiprocessing."""

from multiprocessing import Lock, RawArray
try:
    # python3
    from collections.abc import MutableMapping
except ImportError:
    # python2
    from collections import MutableMapping
import ctypes
import sys

class SharedTable(MutableMapping):
    """Provides a simple shared-memory table of integers, floats, or strings.
    Use this class as follows:

    .. code-block:: python

        tbl = SharedTable({'cnt': 0})
        with tbl.get_lock():
            tbl['startTime'] = time.time()
        for i in range(10):
            with tbl.get_lock():
                tbl['cnt'] += 1
    """

    # currently unused, here for todo below
    types = {
        str: ctypes.c_wchar_p,
        int: ctypes.c_int,
        float: ctypes.c_float
    }

    def __init__(self, init_dict=None):
        """Create a shared memory version of each element of the initial
        dictionary. Creates an empty array otherwise, which will extend
        automatically when keys are added.

        Each different type (all supported types listed in the ``types`` array
        above) has its own array. For each key we store an index into the
        appropriate array as well as the type of value stored for that key.
        """
        # idx is dict of {key: (array_idx, value_type)}
        self.idx = {}
        # arrays is dict of {value_type: array_of_ctype}
        self.arrays = {}
        if init_dict:
            sizes = {typ: 0 for typ in self.types.keys()}
            for v in init_dict.values():
                if type(v) not in sizes:
                    raise TypeError('SharedTable does not support values of ' +
                                    'type ' + str(type(v)))
                sizes[type(v)] += 1
            for typ, sz in sizes.items():
                self.arrays[typ] = RawArray(self.types[typ], sz)
            idxs = {typ: 0 for typ in self.types.keys()}
            for k, v in init_dict.items():
                val_type = type(v)
                self.idx[k] = (idxs[val_type], val_type)
                if val_type == str:
                    v = sys.intern(v)
                self.arrays[val_type][idxs[val_type]] = v
                idxs[val_type] += 1
        # initialize any needed empty arrays
        for typ, ctyp in self.types.items():
            if typ not in self.arrays:
                self.arrays[typ] = RawArray(ctyp, 0)
        self.lock = Lock()

    def __len__(self):
        return sum(len(a) for a in self.arrays.values())

    def __iter__(self):
        return iter(self.idx)

    def __contains__(self, key):
        return key in self.idx

    def __getitem__(self, key):
        """Returns shared value if key is available."""
        if key in self.idx:
            idx, typ = self.idx[key]
            return self.arrays[typ][idx]
        else:
            raise KeyError('Key "{}" not found in SharedTable'.format(key))

    def __setitem__(self, key, value):
        """If key is in table, update it. Otherwise, extend the array to make
        room. This uses additive resizing not multiplicative, since the number
        of keys is not likely to change frequently during a run, so do not abuse
        it.
        Raises an error if you try to change the type of the value stored for
        that key--if you need to do this, you must delete the key first.
        """
        val_type = type(value)
        if val_type not in self.types:
            raise TypeError('SharedTable does not support type ' + type(value))
        if val_type == str:
            value = sys.intern(value)
        if key in self.idx:
            idx, typ = self.idx[key]
            if typ != val_type:
                raise TypeError(('Cannot change stored type for {key} from ' +
                                 '{v1} to {v2}. You need to del the key first' +
                                 ' if you need to change value types.'
                                 ).format(key=key, v1=typ, v2=val_type))
            self.arrays[typ][idx] = value
        else:
            old_array = self.arrays[val_type]
            ctyp = self.types[val_type]
            new_array = RawArray(ctyp, len(old_array) + 1)
            for i in range(len(old_array)):
                new_array[i] = old_array[i]
            new_array[-1] = value
            self.arrays[val_type] = new_array
            self.idx[key] = (len(new_array) - 1, val_type)

    def __delitem__(self, key):
        if key in self.idx:
            idx, typ = self.idx[key]
            old_array = self.arrays[typ]
            new_array = RawArray(self.types[typ], len(old_array) - 1)
            for i in range(len(old_array) - 1):
                new_array[i] = old_array[i]
            self.arrays[typ] = new_array
            del self.idx[key]
        else:
            raise KeyError('Key "{}" not found in SharedTable'.format(key))

    def __str__(self):
        """Returns simple dict representation of the mapping."""
        return '{{{}}}'.format(
            ', '.join(
                '{k}: {v}'.format(k=key, v=self.arrays[typ][idx])
                for key, (idx, typ) in self.idx.items()
            )
        )

    def __repr__(self):
        """Returns the object type and memory location with the mapping."""
        representation = super().__repr__()
        return representation.replace('>', ': {}>'.format(str(self)))

    def get_lock(self):
        return self.lock
