#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Dialogue safety related datasets and teachers.
"""

import json
import os

from parlai.core.teachers import FixedDialogTeacher

from .build import build


class DialogueNliTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None, extras=False):
        super().__init__(opt, shared)

        # Build the data if it doesn't exist.
        build(opt)

        suffix = self.datatype
        if suffix.startswith('train'):
            suffix = 'train'
        elif suffix.startswith('test'):
            suffix = 'test'
        elif suffix.startswith('valid'):
            suffix = 'dev'

        if extras:
            datapath = os.path.join(
                opt['datapath'],
                'dialogue_nli',
                'dnli',
                'dialogue_nli_extra',
                'dialogue_nli_EXTRA_uu_' + suffix + '.jsonl',
            )
        else:
            datapath = os.path.join(
                opt['datapath'],
                'dialogue_nli',
                'dnli',
                'dialogue_nli',
                'dialogue_nli_' + suffix + '.jsonl',
            )

        self._setup_data(datapath)
        self.id = 'dnli'
        self.reset()

    def _setup_data(self, path):
        with open(path) as data_file:
            if 'extra' in path and 'train' in path:
                line = data_file.readline()

                # trim corrupted JSON
                line = line[: line.rfind("{")]
                line = line[: line.rfind(",")] + "]"

                self.data = json.loads(line)
            else:
                self.data = json.load(data_file)

    def num_examples(self):
        return len(self.data)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=0):
        entry = self.data[episode_idx]
        entry['id'] = self.id
        entry['episode_done'] = True
        entry['labels'] = [entry['label']]
        entry['text'] = entry['sentence1'] + '\n' + entry['sentence2']
        return entry


class ExtrasTeacher(DialogueNliTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared, extras=True)


class DefaultTeacher(DialogueNliTeacher):
    pass
