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
    jsonl_dirpath = os.path.join(opt['datapath'], 'redial')
    return jsonl_dirpath


class ReDialTeacher(FixedDialogTeacher):
    """
    ReDial Teacher.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        jsonl_path = _path(opt)
        self._setup_data(jsonl_path)
        self.id = 'redial'
        self.reset()

    def _setup_data(self, jsonl_path):
        test_data = []
        train_data = []
        valid_data = []
        train_path = os.path.join(jsonl_path, 'train_data.jsonl')
        test_path = os.path.join(jsonl_path, 'test_data.jsonl')
        valid_split = 0.5
        if self.datatype.startswith('test'):
            with open(test_path) as f:
                for line in f:
                    test_data.append(json.loads(line))
                self.episodes = test_data
            unmerged_episodes = self.episodes[int(valid_split * len(self.episodes)) :]
        elif self.datatype.startswith('valid'):
            with open(test_path) as f:
                for line in f:
                    valid_data.append(json.loads(line))
                self.episodes = valid_data
            unmerged_episodes = self.episodes[: int(valid_split * len(self.episodes))]
        else:
            with open(train_path) as f:
                for line in f:
                    train_data.append(json.loads(line))
            unmerged_episodes = train_data

        # some speakers speak multiple times in a row.
        self.episodes = []
        for unmerged_episode in unmerged_episodes:
            episode = []
            prev_speaker = None
            for message in unmerged_episode['messages']:
                curr_speaker = message['senderWorkerId']
                if curr_speaker == prev_speaker:
                    episode[-1] = episode[-1] + " " + message['text']
                else:
                    episode.append(message['text'])
                    prev_speaker = curr_speaker
            self.episodes.append(episode)

    def num_examples(self):
        examples = 0
        for data in self.episodes:
            examples += len(data) // 2
        return examples

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=0):
        text_idx = entry_idx * 2
        entry = self.episodes[episode_idx][text_idx]
        episode_done = text_idx == len(self.episodes[episode_idx]) - 2
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.episodes[episode_idx][text_idx + 1]],
        }
        return action


class DefaultTeacher(ReDialTeacher):
    pass
