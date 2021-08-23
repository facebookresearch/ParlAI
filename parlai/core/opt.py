#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Opt is the system for passing around options throughout ParlAI.
"""

from __future__ import annotations

import copy
import json
import pickle
import traceback
import os
import pkg_resources
import parlai.utils.logging as logging

from typing import List

from parlai.utils.io import PathManager

# these keys are automatically removed upon save. This is a rather blunt hammer.
# It's preferred you indicate this at option definition time.
__AUTOCLEAN_KEYS__: List[str] = [
    "override",
    "batchindex",
    "download_path",
    "datapath",
    "verbose",
    # we don't save interactive mode or load from checkpoint, it's only decided by scripts or CLI
    "interactive_mode",
    "load_from_checkpoint",
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
        self.deepcopies = []

    def __setitem__(self, key, val):
        loc = traceback.format_stack(limit=2)[-2]
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
        loc = traceback.format_stack(limit=3)[-3]
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
    def load(cls, optfile: str) -> Opt:
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

    @classmethod
    def load_init(cls, optfile: str) -> Opt:
        """
        Like load, but also looks in opt_presets folders.

        optfile may also be a comma-separated list of multiple presets/files.
        """
        if "," in optfile:
            # load and combine each of the individual files
            new_opt = cls()
            for subopt in optfile.split(","):
                new_opt.update(cls.load_init(subopt))
            return new_opt

        oa_filename = os.path.join("opt_presets", optfile + ".opt")
        user_filename = os.path.join(os.path.expanduser(f"~/.parlai"), oa_filename)
        if PathManager.exists(optfile):
            return cls.load(optfile)
        elif PathManager.exists(user_filename):
            # use a user's custom opt preset
            return cls.load(user_filename)
        else:
            # Maybe a bundled opt preset
            for root in ['parlai', 'parlai_internal', 'parlai_fb']:
                try:
                    if pkg_resources.resource_exists(root, oa_filename):
                        return cls.load(
                            pkg_resources.resource_filename(root, oa_filename)
                        )
                except ModuleNotFoundError:
                    continue

        # made it through without a return path so raise the error
        raise FileNotFoundError(
            f"Could not find filename '{optfile} or opt preset '{optfile}.opt'. "
            "Please check https://parl.ai/docs/opt_presets.html for a list "
            "of available opt presets."
        )

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
