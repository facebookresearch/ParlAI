#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Opt is the system for passing around options throughout ParlAI.
"""

import copy
import json
import pickle
import traceback


class Opt(dict):
    """
    Class for tracking options.

    Functions like a dict, but allows us to track the history of arguments as they are
    set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.deepcopies = []

    def __setitem__(self, key, val):
        loc = traceback.format_stack()[-2]
        self.history.append((key, val, loc))
        super().__setitem__(key, val)

    def __getstate__(self):
        return (self.history, self.deepcopies, dict(self))

    def __setstate__(self, state):
        self.history, self.deepcopies, data = state
        self.update(data)

    def __reduce__(self):
        return (Opt, (), self.__getstate__())

    def __deepcopy__(self, memo):
        """
        Override deepcopy so that history is copied over to new object.
        """
        # track location of deepcopy
        loc = traceback.format_stack()[-3]
        self.deepcopies.append(loc)
        # copy all our children
        memo = Opt({k: copy.deepcopy(v) for k, v in self.items()})
        # deepcopy the history. history is only tuples, so we can do it shallow
        memo.history = copy.copy(self.history)
        # deepcopy the list of deepcopies. also shallow bc only strings
        memo.deepcopies = copy.copy(self.deepcopies)
        return memo

    def display_deepcopies(self):
        """
        Display all deepcopies.
        """
        if len(self.deepcopies) == 0:
            return 'No deepcopies performed on this opt.'
        return '\n'.join(f'{i}. {loc}' for i, loc in enumerate(self.deepcopies, 1))

    def display_history(self, key):
        """
        Display the history for an item in the dict.
        """
        changes = []
        i = 0
        for key_, val, loc in self.history:
            if key != key_:
                continue
            i += 1
            changes.append(f'{i}. {key} was set to {val} at:\n{loc}')
        if changes:
            return '\n'.join(changes)
        else:
            return f'No history for {key}'


def load_opt_file(optfile: str) -> Opt:
    """
    Load an Opt from disk.
    """
    try:
        # try json first
        with open(optfile, 'r') as t_handle:
            opt = json.load(t_handle)
    except UnicodeDecodeError:
        # oops it's pickled
        with open(optfile, 'rb') as b_handle:
            opt = pickle.load(b_handle)
    return Opt(opt)
