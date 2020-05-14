#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
import re
import csv


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    jsonl_dirpath = os.path.join(opt['datapath'], 'redial')
    return jsonl_dirpath


# Turns title from format "Title (Year)" to "Title" or leaves as is if no (Year)
def remove_year_from_title(title):
    matches = re.finditer(r"\s\(", title)
    indices = [m.start(0) for m in matches]
    if indices:
        title_end = indices[-1]
        return title[:title_end]
    else:
        return title


def replace_movie_ids(id_string, id_map):
    pattern = r'@\d+'
    return re.sub(pattern, lambda s: id_map[s.group()], id_string)


class ReDialTeacher(FixedDialogTeacher):
    """
    ReDial Teacher.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        jsonl_path = _path(opt)
        self.title_id_map = {}
        self.get_title_dict(jsonl_path)
        self._setup_data(jsonl_path)
        self.id = 'redial'

        self.reset()

    def get_title_dict(self, path):
        csv_path = os.path.join(path, 'movies_with_mentions.csv')
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.title_id_map['@' + row[0]] = remove_year_from_title(row[1])

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
                text = replace_movie_ids(message['text'], self.title_id_map)
                if curr_speaker == prev_speaker:
                    episode[-1] = episode[-1] + " " + text
                else:
                    episode.append(text)
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
        final_speaker_idx = len(self.episodes[episode_idx]) - 2
        episode_done = text_idx >= final_speaker_idx
        if text_idx < final_speaker_idx:
            labels = [self.episodes[episode_idx][text_idx + 1]]
        else:
            labels = ['']
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': labels,
        }
        return action


class DefaultTeacher(ReDialTeacher):
    pass
