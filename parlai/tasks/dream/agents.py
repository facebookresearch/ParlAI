#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json


class DREAMTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'dream'
        if shared is not None:
            self.episodes = shared['episodes']
        else:
            self.episodes = self.setup_data()
        self.reset()

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def setup_data(self):
        build(self.opt)
        jsons_path = os.path.join(self.opt['datapath'], 'DREAM')
        return self.setup_helper(jsons_path)

    def setup_helper(self, jsons_path):
        if self.opt['datatype'].startswith('test'):
            dpath = os.path.join(jsons_path, 'test.json')
        elif self.opt['datatype'].startswith('valid'):
            dpath = os.path.join(jsons_path, 'dev.json')
        elif self.opt['datatype'].startswith('train'):
            dpath = os.path.join(jsons_path, 'train.json')
        else:
            raise ValueError('Datatype not train, test, or valid')
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


class DefaultTeacher(DREAMTeacher):
    pass
