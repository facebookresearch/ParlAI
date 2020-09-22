#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from .build import build
from parlai.utils.io import PathManager
import os
import json


class CommonSenseQATeacher(FixedDialogTeacher):
    """
    CommonSenseQA is a multiple-choice Q-A dataset that relies on commonsense knowlegde
    to predict correct answers. More information found at:

    <https://www.tau-nlp.org/commonsenseqa>.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'commonsenseqa'
        build(opt)
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
        jsons_path = os.path.join(self.opt['datapath'], 'CommonSenseQA')
        dtype = self.opt['datatype']
        if dtype.startswith('test'):
            dpath = os.path.join(jsons_path, 'dev.jsonl')
        elif dtype.startswith('train') or dtype.startswith('valid'):
            dpath = os.path.join(jsons_path, 'train.jsonl')
        else:
            raise ValueError('Datatype not train, test, or valid')
        episodes = []
        with PathManager.open(dpath) as f:
            for line in f:
                episodes.append(json.loads(line))
        # There are 1221 episodes in the test set. Making the valid set this
        # large will make about an 80/10/10 split, as the paper had for splits.
        test_set_episodes = 1221
        if dtype.startswith('valid'):
            episodes = episodes[:test_set_episodes]
        elif dtype.startswith('train'):
            episodes = episodes[test_set_episodes:]
        return episodes

    def num_examples(self):
        return len(self.episodes)

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=0):
        episode = self.episodes[episode_idx]
        answer = episode['answerKey']
        candidates = []
        for choice in episode['question']['choices']:
            if choice['label'] == answer:
                labels = [choice['text']]
            candidates.append(choice['text'])
        action = {
            'id': self.id,
            'text': episode['question']['stem'],
            'episode_done': True,
            'labels': labels,
            'label_candidates': candidates,
        }
        return action


class DefaultTeacher(CommonSenseQATeacher):
    pass
