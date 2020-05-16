#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    jsons_path = os.path.join(opt['datapath'], 'DREAM')
    return jsons_path


class BaseMultipleChoiceTeacher(FixedDialogTeacher):
    """
    Base class for Dream and C3 Teachers.
    """

    def __init__(self, opt, path_fn, shared=None):
        super().__init__(opt, shared)
        jsons_path = path_fn(opt)
        self.episodes = self._setup_data(jsons_path)
        self.reset()

    def _setup_data(self, jsons_path):
        if self.opt['datatype'].startswith('test'):
            dpath = os.path.join(jsons_path, 'test.json')
        elif self.opt['datatype'].startswith('valid'):
            dpath = os.path.join(jsons_path, 'dev.json')
        else:
            dpath = os.path.join(jsons_path, 'train.json')
        episodes = []
        with open(dpath) as f:
            data = json.load(f)
            for dialogue in data:
                context = '\n'.join(dialogue[0])
                qas = dialogue[1]
                episodes.append({'context': context, 'qas': qas})
        return episodes

    def num_examples(self):
        examples = 0
        for data in self.episodes:
            examples += len(data['qas'])
        return examples

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=0):
        episode = self.episodes[episode_idx]
        entry = episode['qas'][entry_idx]['question']
        if entry_idx == 0:
            entry = episode['context'] + '\n' + entry
        episode_done = entry_idx == len(episode['qas']) - 1
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [episode['qas'][entry_idx]['answer']],
            'label_candidates': episode['qas'][entry_idx]['choice'],
        }
        return action


class DREAMTeacher(BaseMultipleChoiceTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, _path, shared)
        self.id = 'dream'


class DefaultTeacher(DREAMTeacher):
    pass
