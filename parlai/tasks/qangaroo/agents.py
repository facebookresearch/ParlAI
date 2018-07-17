# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import DialogTeacher, FixedDialogTeacher
from .build import build

import json
import os

class IndexTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        datapath = self._path(opt)
        super().__init__(opt, shared)
        self.data = self._setup_data(datapath)
        self.id = 'qangaroo'
        self.reset()

    def _path(self, opt):
        # Build the data if it doesn't exist.
        build(opt)
        dt = opt['datatype'].split(':')
        datatype = 'dev' if dt[0] == 'valid' else 'train'
        return os.path.join(opt['datapath'], 'qangaroo', 'qangaroo_v1.1',
                            'wikihop', datatype + '.json')

    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        item = self.examples[episode_idx]
        action = {
            'id': 'qangaroo',
            'text': '\n'.join(item['supports']),
            'query': item['query'],
            'label_candidates': item['candidates'],
            'label': item['answer'],
            'supports': item['supports'],
            'episode_done': True,
        }
        return action

    def _setup_data(self, path):
        print("loading: ", path)
        with open(path) as data_file:
            self.examples = json.load(data_file)


class WikiHopTeacher(IndexTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def _path(self, opt):
        # Build the data if it doesn't exist.
        build(opt)
        dt = opt['datatype'].split(':')
        datatype = 'dev' if dt[0] == 'valid' else 'train'
        return os.path.join(opt['datapath'], 'qangaroo', 'qangaroo_v1.1',
                            'wikihop', datatype + '.json')


class MaskedWikiHopTeacher(IndexTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def _path(self, opt):
        # Build the data if it doesn't exist.
        build(opt)
        dt = opt['datatype'].split(':')
        datatype = 'dev.masked' if dt[0] == 'valid' else 'train.masked'
        return os.path.join(opt['datapath'], 'qangaroo', 'qangaroo_v1.1',
                            'wikihop', datatype + '.json')


class MedHopTeacher(IndexTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def _path(self, opt):
        # Build the data if it doesn't exist.
        build(opt)
        dt = opt['datatype'].split(':')
        datatype = 'dev' if dt[0] == 'valid' else dt[0]
        return os.path.join(opt['datapath'], 'qangaroo', 'qangaroo_v1.1',
                            'medhop', datatype + '.json')


class MaskedMedHopTeacher(IndexTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def _path(self, opt):
        # Build the data if it doesn't exist.
        build(opt)
        dt = opt['datatype'].split(':')
        datatype = 'dev.masked' if dt[0] == 'valid' else 'train.masked'
        return os.path.join(opt['datapath'], 'qangaroo', 'qangaroo_v1.1',
                            'medhop', datatype + '.json')
