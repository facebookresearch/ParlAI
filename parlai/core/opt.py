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
        self.history = {}
        self.deepcopies = []

    def __setitem__(self, key, val):
        loc = traceback.format_stack()[-2]
        self.history.setdefault(key, []).append((loc, val))
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
        # deepcopy the dict
        memo = copy.deepcopy(dict(self))
        # make into Opt object
        memo = Opt(memo)
        # deepcopy the history
        memo.history = copy.deepcopy(self.history)
        # deepcopy the deepcopy history
        memo.deepcopies = copy.deepcopy(self.deepcopies)
        return memo

    def display_deepcopies(self):
        """
        Display all deepcopies.
        """
        if len(self.deepcopies) == 0:
            print('No deepcopies performed on this opt.')
            return
        print('Deepcopies were performed at the following locations:\n')
        for i, loc in enumerate(self.deepcopies):
            print('{}. {}'.format(i + 1, loc))

    def display_history(self, key):
        """
        Display the history for an item in the dict.
        """
        if key not in self.history:
            print('No history for key {}.'.format(key))
            return
        item_hist = self.history[key]
        for i, change in enumerate(item_hist):
            print(
                '{}. {} was set to {} at:\n{}\n'.format(
                    i + 1, key, change[1], change[0]
                )
            )


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
