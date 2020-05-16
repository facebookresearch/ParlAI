#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json


def setup_data(opt, jsons_path):
    if opt['datatype'].startswith('test'):
        dpath = os.path.join(jsons_path, 'test.json')
    elif opt['datatype'].startswith('valid'):
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


def num_examples(episodes):
    examples = 0
    for data in episodes:
        examples += len(data['qas'])
    return examples


def get(tid, episodes, episode_idx, entry_idx=0):
    episode = episodes[episode_idx]
    entry = episode['qas'][entry_idx]['question']
    if entry_idx == 0:
        entry = episode['context'] + '\n' + entry
    episode_done = entry_idx == len(episode['qas']) - 1
    action = {
        'id': tid,
        'text': entry,
        'episode_done': episode_done,
        'labels': [episode['qas'][entry_idx]['answer']],
        'label_candidates': episode['qas'][entry_idx]['choice'],
    }
    return action


class DREAMTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        build(opt)
        jsons_path = os.path.join(opt['datapath'], 'DREAM')
        self.id = 'dream'
        self.episodes = setup_data(opt, jsons_path)
        self.reset()

    def num_examples(self):
        return num_examples(self.episodes)

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=0):
        return get(self.id, self.episodes, entry_idx)


class DefaultTeacher(DREAMTeacher):
    pass
