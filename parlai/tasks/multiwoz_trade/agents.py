#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.utils.io import PathManager
from .build import build
from .utils.trade_proc import trade_process
from .utils.reformat import reformat_dst
import os
import json, random


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    data_dir = os.path.join(opt['datapath'], 'multiwoz_trade', 'MULTIWOZ2.1')
    conversations_path = os.path.join(data_dir, 'data.json')
    return conversations_path, data_dir


class MultiWozTeacher(FixedDialogTeacher):
    """
    MultiWOZ 2.1 Teacher.

    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], jsons_path = _path(opt)
        self._setup_data(opt['datafile'], jsons_path)
        self.id = 'multiwoz_trade'
        self.reset()

    def _setup_data(self, data_path, jsons_path):
        print('loading: ' + data_path)
        with PathManager.open(data_path) as data_file:
            self.messages = json.load(data_file)

        test_path = os.path.join(jsons_path, 'testListFile.json')
        valid_path = os.path.join(jsons_path, 'valListFile.json')
        if self.datatype.startswith('test'):
            with PathManager.open(test_path) as f:
                test_data = {line.strip(): self.messages[line.strip()] for line in f}
                self.messages = test_data
        elif self.datatype.startswith('valid'):
            with PathManager.open(valid_path) as f:
                valid_data = {line.strip(): self.messages[line.strip()] for line in f}
                self.messages = valid_data
        else:
            with PathManager.open(test_path) as f:
                for line in f:
                    if line.strip() in self.messages:
                        del self.messages[line.strip()]
            with PathManager.open(valid_path) as f:
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


class MultiWozDSTTeacher(FixedDialogTeacher):
    """
    MultiWOZ2.1 DST Teacher.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'multiwoz_trade'

        # # # reading args
        self.just_test = opt.get('just_test', False)
        self.seed = opt.get('rand_seed', 0)

        # # # set random seeds
        random.seed(self.seed)

        opt['datafile'], data_dir = self._path(opt)
        self._setup_data(opt['datafile'], data_dir)

        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('MultiWozDST Teacher Args')
        agent.add_argument(
            '--just_test',
            type='bool',
            default=False,
            help="True if one would like to test agent's training with small amount of data (default: False).",
        )
        agent.add_argument(
            '--rand_seed',
            type=int,
            default=0,
            help="specify to set random seed (default: 0).",
        )

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _path(self, opt):
        # set up path to data (specific to each dataset)
        data_dir = os.path.join(opt['datapath'], 'multiwoz_trade', 'MULTIWOZ2.1')
        data_path = os.path.join(data_dir, 'data_reformat_trade_dst.json')

        # build the data if it does not exist
        build(opt)

        # process the data with TRADE's code, if it does not exist
        if not os.path.exists(os.path.join(data_dir, 'dials_trade.json')):
            trade_process(data_dir)

        # reformat data for DST
        if not os.path.exists(data_path):
            reformat_dst(data_dir)

        return data_path, data_dir

    def _setup_data(self, data_path, jsons_path):
        # # # loading directly from test file or val file
        if self.datatype.startswith('test'):
            test_path = data_path.replace(".json", "_test.json")
            test_data = self._load_json(test_path)
            self.messages = list(test_data.values())
        elif self.datatype.startswith('valid'):
            valid_path = data_path.replace(".json", "_valid.json")
            valid_data = self._load_json(valid_path)
            self.messages = list(valid_data.values())
        else:
            train_path = data_path.replace(".json", "_train.json")
            train_data = self._load_json(train_path)
            self.messages = list(train_data.values())
            random.shuffle(self.messages)
        
        if self.just_test:
            self.messages = self.messages[:100]


    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        domains    = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]

        slot_types = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
                    "area", "leave", "stars", "department", "people", "time", "food", 
                    "post", "phone", "name", 'internet', 'parking',
                    'book stay', 'book people','book time', 'book day',
                    'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']
        slots_list = []

        # # # split according to ","
        str_split = slots_string.strip().split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in domains:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1]+" "+slot[2]
                    slot_val  = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val  = " ".join(slot[2:])
                if not slot_val == 'dontcare':
                    slots_list.append(domain+"--"+slot_type+"--"+slot_val)
        return slots_list

    def custom_evaluation(self, teacher_action: Message, labels, model_response: Message):
        resp = model_response.get('text')
        if not resp:
            return

        # # # extract ground truth from labels
        slots_truth = self._extract_slot_from_string(labels[0])
        
        # # # extract generated slots from model_response
        slots_pred = self._extract_slot_from_string(resp)

        self.metrics.add('joint goal acc', AverageMetric(set(slots_truth) == set(slots_pred)))

    def num_examples(self):
        # each turn be seen as a individual dialog
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        # log_idx = entry_idx
        entry = self.messages[episode_idx]['context']
        episode_done = True
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx]['slots_inf']],
            'dial_id': self.messages[episode_idx]['dial_id'],
            'turn_num': self.messages[episode_idx]['turn_num'],
        }
        return action

class DefaultTeacher(MultiWozDSTTeacher):
    pass
