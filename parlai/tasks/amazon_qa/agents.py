#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build, RESOURCES
import os
import json


EXISTING_KEYS = [
    'question',
    'answer',
    'asin',
    'questionType',
    'questionTime',
    'askerID',
    'answerType',
    'answerTime',
    'unixTime',
    'answererID',
    'helpful',
    'answerScore',
]


class DefaultTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        super().__init__(opt, shared)

        dt = opt['datatype'].split(':')[0]
        if dt != 'train':
            raise RuntimeError('Not valid datatype (only train).')

        if shared:
            self.data = shared['data']
            self.num_ex = shared['num_ex']
            self.num_ep = shared['num_ep']
        else:
            build(opt)
            self._setup_data()
            self.num_ex = sum([len(x) for x in self.data])
            self.num_ep = len(self.data)
        self.reset()

    def num_episodes(self):
        return self.num_ep

    def num_examples(self):
        return self.num_ex

    def _setup_data(self):
        self.data = []

        def create_entry_single(episode):
            entry = []
            for key in EXISTING_KEYS:
                if key in episode:
                    entry.append(str(episode[key]))
                else:
                    entry.append('N/A')
            return entry

        def create_entry_multiple(episode):
            entries = []

            for question in episode['questions']:
                new_episode = dict()
                new_episode['asin'] = episode['asin']
                new_episode['askerID'] = question['askerID']
                new_episode['questionTime'] = question['questionTime']
                new_episode['quesitonType'] = question['questionType']
                new_episode['question'] = question['questionText']

                for answer in question['answers']:
                    answer.update(new_episode)
                    answer['answer'] = answer['answerText']
                    entries.append([create_entry_single(answer)])

            return entries

        fpath = os.path.join(self.opt['datapath'], 'AmazonQA')
        for i, f in enumerate(RESOURCES):
            json_file = f.file_name[:-3]
            file_path = os.path.join(fpath, json_file)

            with PathManager.open(file_path, 'r') as infile:
                data = infile.read()
                new_data = data.replace('}\n{', '},\n{')
                json_data = json.loads(f'[{new_data}]')

            for ep in json_data:
                # First 20 datasets have a different format than those later
                if i < 21:
                    self.data.append([create_entry_single(ep)])
                else:
                    self.data += create_entry_multiple(ep)

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        entry = ep[entry_idx]
        action = dict()
        action['id'] = episode_idx
        for i, key in enumerate(EXISTING_KEYS):
            if i < 2:
                continue
            action[key] = entry[i]
        action['episode_done'] = True
        action['text'] = entry[0]
        action['labels'] = [entry[1]]

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['num_ex'] = self.num_ex
        shared['num_ep'] = self.num_ep
        return shared
