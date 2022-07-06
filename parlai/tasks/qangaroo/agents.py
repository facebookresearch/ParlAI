#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build

import json
import os


class DefaultTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        # Build the data if it doesn't exist.
        build(opt)
        if not shared:
            datapath = self._path(opt)
            self._setup_data(datapath)
        else:
            self.examples = shared['examples']
        self.id = 'qangaroo'
        self.reset()

    def _path(self, opt):
        dt = opt['datatype'].split(':')
        datatype = 'train' if dt[0] == 'train' else 'dev'
        return os.path.join(
            opt['datapath'], 'qangaroo', 'qangaroo_v1.1', 'wikihop', datatype + '.json'
        )

    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        # same as num_examples since only one exchange per episode
        return self.num_examples()

    def share(self):
        shared = super().share()
        shared['examples'] = self.examples
        return shared

    def get(self, episode_idx, entry_idx=None):
        item = self.examples[episode_idx]
        action = {
            'id': 'qangaroo',
            'text': '\n'.join(item['supports']) + '\n' + item['query'],
            'query': item['query'],
            'label_candidates': item['candidates'],
            'labels': [item['answer']],
            'supports': item['supports'],
            'episode_done': True,
        }
        return action

    def _setup_data(self, path):
        print("loading: ", path)
        with PathManager.open(path) as data_file:
            self.examples = json.load(data_file)


class WikiHopTeacher(DefaultTeacher):
    def _path(self, opt):
        dt = opt['datatype'].split(':')
        datatype = 'train' if dt[0] == 'train' else 'dev'
        return os.path.join(
            opt['datapath'], 'qangaroo', 'qangaroo_v1.1', 'wikihop', datatype + '.json'
        )


class MaskedWikiHopTeacher(DefaultTeacher):
    def _path(self, opt):
        dt = opt['datatype'].split(':')
        datatype = 'train.masked' if dt[0] == 'train' else 'dev.masked'
        return os.path.join(
            opt['datapath'], 'qangaroo', 'qangaroo_v1.1', 'wikihop', datatype + '.json'
        )


class MedHopTeacher(DefaultTeacher):
    def _path(self, opt):
        dt = opt['datatype'].split(':')
        datatype = 'train' if dt[0] == 'train' else 'dev'
        return os.path.join(
            opt['datapath'], 'qangaroo', 'qangaroo_v1.1', 'medhop', datatype + '.json'
        )


class MaskedMedHopTeacher(DefaultTeacher):
    def _path(self, opt):
        dt = opt['datatype'].split(':')
        datatype = 'train.masked' if dt[0] == 'train' else 'dev.masked'
        return os.path.join(
            opt['datapath'], 'qangaroo', 'qangaroo_v1.1', 'medhop', datatype + '.json'
        )
