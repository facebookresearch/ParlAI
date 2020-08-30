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
import parlai.utils.logging as logging

from typing import List

from parlai.utils.io import PathManager

# these keys are automatically removed upon save. This is a rather blunt hammer.
# It's preferred you indicate this at option definiton time.
__AUTOCLEAN_KEYS__: List[str] = [
    "override",
    "batchindex",
    "download_path",
    "datapath",
    "batchindex",
    # we don't save interactive mode, it's only decided by scripts or CLI
    "interactive_mode",
]


class Opt(dict):
    """
    Class for tracking options.

    Functions like a dict, but allows us to track the history of arguments as they are
    set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def __setitem__(self, key, val):
        raise RuntimeError(
            'Setting values in opt is no longer allowed. '
            'Use opt = opt.fork(key=newvalue).'
        )

    def update(self, *args, **kwargs):
        raise RuntimeError(
            'Setting values in opt is no longer allowed. '
            'Use opt = opt.fork(key=newvalue).'
        )

    def todo__del__(self, key):
        pass

    def __getstate__(self):
        return self.history, dict(self)

    def __setstate__(self, state):
        self.history, data = state
        super().update(data)

    def __reduce__(self):
        return (Opt, (), self.__getstate__())

    def fork(self, **newvalues):
        loc = traceback.format_stack(limit=2)[-2]
        copied = dict(self)
        newhistory = copy.copy(self.history)
        for key, value in newvalues.items():
            copied[key] = value
            newhistory.append((key, value, loc))
        retval = Opt(copied)
        retval.history = newhistory
        return retval

    def __deepcopy__(self, memo):
        """
        Override deepcopy so that history is copied over to new object.
        """
        # track location of deepcopy
        # copy all our children
        memo = Opt({k: copy.deepcopy(v) for k, v in self.items()})
        # deepcopy the history. history is only tuples, so we can do it shallow
        memo.history = self.history[:]
        return memo

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

    def save(self, filename: str) -> None:
        """
        Save the opt to disk.

        Attempts to 'clean up' any residual values automatically.
        """
        # start with a shallow copy
        dct = dict(self)

        # clean up some things we probably don't want to save
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]

        with PathManager.open(filename, 'w', encoding='utf-8') as f:
            json.dump(dct, fp=f, indent=4)
            # extra newline for convenience of working with jq
            f.write('\n')

    @classmethod
    def load(cls, optfile: str) -> 'Opt':
        """
        Load an Opt from disk.
        """
        try:
            # try json first
            with PathManager.open(optfile, 'r', encoding='utf-8') as t_handle:
                dct = json.load(t_handle)
        except UnicodeDecodeError:
            # oops it's pickled
            with PathManager.open(optfile, 'rb') as b_handle:
                dct = pickle.load(b_handle)
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]
        return cls(dct)

    def log(self, header="Opt"):
        from parlai.core.params import print_git_commit

        logging.info(header + ":")
        for key in sorted(self.keys()):
            valstr = str(self[key])
            if valstr.replace(" ", "").replace("\n", "") != valstr:
                # show newlines as escaped keys, whitespace with quotes, etc
                valstr = repr(valstr)
            logging.info(f"    {key}: {valstr}")
        print_git_commit()
