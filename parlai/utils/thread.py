#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provides utilities useful for multiprocessing.

This includes a ``SharedTable``.
"""

from multiprocessing import Lock
from multiprocessing import RawArray  # type: ignore
from collections.abc import MutableMapping
import ctypes
import sys


class SharedTable(MutableMapping):
    """
    Provides a simple shared-memory table of integers, floats, or strings.

    Use this class as follows:

    .. code-block:: python

        tbl = SharedTable({'cnt': 0})
        with tbl.get_lock():
            tbl['startTime'] = time.time()
        for i in range(10):
            with tbl.get_lock():
                tbl['cnt'] += 1
    """

    types = {int: ctypes.c_int, float: ctypes.c_float, bool: ctypes.c_bool}

    def __init__(self, init_dict=None):
        """
        Create a shared memory version of each element of the initial dictionary.

        Creates an empty array otherwise, which will extend
        automatically when keys are added.

        Each different type (all supported types listed in the ``types`` array
        above) has its own array. For each key we store an index into the
        appropriate array as well as the type of value stored for that key.
        """
        # idx is dict of {key: (array_idx, value_type)}
        self.idx = {}
        # arrays is dict of {value_type: array_of_ctype}
        self.arrays = {}
        self.tensors = {}

        if init_dict:
            sizes = {typ: 0 for typ in self.types.keys()}
            for k, v in init_dict.items():
                if is_tensor(v):
                    # add tensor to tensor dict--don't try to put in rawarray
                    self.tensors[k] = v
                    continue
                elif type(v) not in sizes:
                    raise TypeError(
                        'SharedTable does not support values of '
                        + 'type '
                        + str(type(v))
                    )
                sizes[type(v)] += 1
            # pop tensors from init_dict
            for k in self.tensors.keys():
                init_dict.pop(k)
            # create raw arrays for each type
            for typ, sz in sizes.items():
                self.arrays[typ] = RawArray(self.types[typ], sz)
            # track indices for each key, assign them to their typed rawarray
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
        return len(self.idx) + len(self.tensors)

    def __iter__(self):
        return iter([k for k in self.idx] + [k for k in self.tensors])

    def __contains__(self, key):
        return key in self.idx or key in self.tensors

    def __getitem__(self, key):
        """
        Return shared value if key is available.
        """
        if key in self.tensors:
            return self.tensors[key]
        elif key in self.idx:
            idx, typ = self.idx[key]
            return self.arrays[typ][idx]
        else:
            raise KeyError('Key "{}" not found in SharedTable'.format(key))

    def __setitem__(self, key, value):
        """
        If key is in table, update it. Otherwise, extend the array to make room.

        This uses additive resizing not multiplicative, since the number
        of keys is not likely to change frequently during a run, so do not
        abuse it.

        Raises an error if you try to change the type of the value stored for
        that key -- if you need to do this, you must delete the key first.
        """
        val_type = type(value)
        if 'Tensor' in str(val_type):
            self.tensors[key] = value
            return
        if val_type not in self.types:
            raise TypeError('SharedTable does not support type ' + str(type(value)))
        if val_type == str:
            value = sys.intern(value)
        if key in self.idx:
            idx, typ = self.idx[key]
            if typ != val_type:
                raise TypeError(
                    (
                        'Cannot change stored type for {key} from '
                        + '{v1} to {v2}. You need to del the key first'
                        + ' if you need to change value types.'
                    ).format(key=key, v1=typ, v2=val_type)
                )
            self.arrays[typ][idx] = value
        else:
            raise KeyError(
                'Cannot add more keys to the shared table as '
                'they will not be synced across processes.'
            )

    def __delitem__(self, key):
        if key in self.tensors:
            del self.tensors[key]
        elif key in self.idx:
            del self.idx[key]
        else:
            raise KeyError('Key "{}" not found in SharedTable'.format(key))

    def __str__(self):
        """
        Return simple dict representation of the mapping.
        """
        lhs = [
            '{k}: {v}'.format(k=key, v=self.arrays[typ][idx])
            for key, (idx, typ) in self.idx.items()
        ]
        rhs = ['{k}: {v}'.format(k=k, v=v) for k, v in self.tensors.items()]
        return '{{{}}}'.format(', '.join(lhs + rhs))

    def __repr__(self):
        """
        Return the object type and memory location with the mapping.
        """
        representation = super().__repr__()
        return representation.replace('>', ': {}>'.format(str(self)))

    def get_lock(self):
        """
        Return the lock.
        """
        return self.lock


def is_tensor(v):
    """
    Return if an object is a torch Tensor, without importing torch.
    """
    if type(v).__module__.startswith('torch'):
        import torch

        return torch.is_tensor(v)
    return False
