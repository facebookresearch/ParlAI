#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher, FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build

import csv
import random
import os


class UbuntuTeacher(DialogTeacher):
    """
    This teacher inherits from the core Dialog Teacher, which just requires it to define
    an iterator over its data `setup_data` in order to inherit basic metrics, a default
    `act` function, and enables Hogwild training with shared memory with no extra work.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'], 'Ubuntu', opt['datatype'].split(':')[0] + '.csv'
        )
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with PathManager.open(path, 'r', newline='') as read:
            csv_read = csv.reader(read)
            next(csv_read)  # eat header

            for line in csv_read:
                fields = [
                    s.replace('__eou__', '.').replace('__eot__', '\n').strip()
                    for s in line
                ]
                context = fields[0]
                response = fields[1]
                cands = None
                if len(fields) > 3:
                    cands = [fields[i] for i in range(2, len(fields))]
                    cands.append(response)
                    random.shuffle(cands)
                yield (context, [response], None, cands), True


class MultiturnTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        opt['datafile'] = os.path.join(
            opt['datapath'], 'Ubuntu', opt['datatype'].split(':')[0] + '.csv'
        )
        super().__init__(opt, shared)

        if shared:
            self.data = shared['data']
        else:
            build(opt)
            fold = opt.get('datatype', 'train').split(':')[0]
            self._setup_data(fold)

        self.num_exs = sum(len(d) for d in self.data)

        # we learn from both sides of every conversation
        self.num_eps = 2 * len(self.data)
        self.reset()

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, fold):
        self.data = []
        fpath = os.path.join(self.opt['datapath'], 'Ubuntu', fold + '.csv')
        print('loading: ' + fpath)
        with PathManager.open(fpath, 'r', newline='') as read:
            csv_read = csv.reader(read)
            next(csv_read)  # eat header

            for line in csv_read:
                fields = line[0].strip().split('__eou__ __eot__')
                fields.append(line[1].strip())
                dialog = []
                for field in fields:
                    if field != '':
                        dialog.append(field.replace('__eou__', '.').strip())
                if len(dialog) > 2:
                    self.data.append(dialog)

    def get(self, episode_idx, entry_idx=0):
        # Sometimes we're speaker 1 and sometimes we're speaker 2
        speaker_id = episode_idx % 2
        full_eps = self.data[episode_idx // 2]

        entries = full_eps
        their_turn = entries[speaker_id + 2 * entry_idx]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]

        episode_done = 1 + speaker_id + 2 * entry_idx >= len(full_eps) - 2

        action = {'text': their_turn, 'labels': [my_turn], 'episode_done': episode_done}
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DefaultTeacher(UbuntuTeacher):
    pass
