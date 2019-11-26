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
    jsons_path = os.path.join(opt['datapath'], 'multiwoz', 'MULTIWOZ2.1')
    conversations_path = os.path.join(jsons_path, 'data.json')
    return conversations_path, jsons_path


class MultiWozTeacher(FixedDialogTeacher):
    """
    MultiWOZ Teacher.

    This dataset contains more than just dialogue. It also contains:
    data.json also contains the following information:
    1. information about the goal of the conversation (what the person is looking for)
    2. topic of the conversation
    3. log of the conversation (the conversation itself + metadata about each utterance)
          Metadata: any taxi, police, restaurant, hospital, hotel, attraction, or train info mentioned
    Information about each metadata category is also contained in its own json file.
    1. restaurant.json: Restaurants + their attributes in the Cambridge UK area (address, pricerange, food...)
    2. attraction.json: Attractions + their attributes in the Cambridge UK area (location, address, entrance fee...)
    3. hotel_db.json: Hotels + their attributes in the Cambridge UK area  (address, price, name, stars...)
    4. train_db.json: Trains + their attributes in the Cambridge UK area (destination, price, departure...)
    5. hospital_db.json: data about the Cambridge hospital's departments (department, phone, id)
    6. police_db.json: Name address and phone number of Cambridge police station
    7. taxi_db.json: Taxi information (taxi_colors, taxi_types, taxi_phone)
    More information about the jsons can be found in readme.json
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], jsons_path = _path(opt)
        self._setup_data(opt['datafile'], jsons_path)
        self.id = 'multiwoz'
        self.reset()

    def _setup_data(self, data_path, jsons_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.messages = json.load(data_file)

        test_path = os.path.join(jsons_path, 'testListFile.json')
        valid_path = os.path.join(jsons_path, 'valListFile.json')
        if self.datatype.startswith('test'):
            with open(test_path) as f:
                test_data = {line.strip(): self.messages[line.strip()] for line in f}
                self.messages = test_data
        elif self.datatype.startswith('valid'):
            with open(valid_path) as f:
                valid_data = {line.strip(): self.messages[line.strip()] for line in f}
                self.messages = valid_data
        else:
            with open(test_path) as f:
                for line in f:
                    if line.strip() in self.messages:
                        del self.messages[line.strip()]
            with open(valid_path) as f:
                for line in f:
                    if line.strip() in self.messages:
                        del self.messages[line.strip()]
        self.messages = list(self.messages.values())

    def num_examples(self):
        examples = 0
        for data in self.messages:
            examples += len(data['log']) // 2
        return examples

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        log_idx = entry_idx * 2
        entry = self.messages[episode_idx]['log'][log_idx]['text']
        episode_done = log_idx == len(self.messages[episode_idx]['log']) - 2
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx]['log'][log_idx + 1]['text']],
        }
        return action


class DefaultTeacher(MultiWozTeacher):
    pass
