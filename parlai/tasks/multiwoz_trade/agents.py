#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from .build import build
import os
import json
import random
from parlai.tasks.dialog_blender.blender import Blender
from parlai.core.worlds import create_task
import numpy as np



class MultiWozTradeTeacher(FixedDialogTeacher):
    """
    MultiWOZ 2.1 Teacher with TRADE preprocess
    Two main differences from multiwoz_v2.1:
    1. with TRADE preprocess, which would normalize and split data
    into "_train.json" "_val.json" and "_test.json"
    2. each turn is an individual episode, turn["context"] includes
    all the dialog history in previous turns.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'multiwoz_trade'
        opt['datafile'], data_dir = self._path(opt)
        self._setup_data(opt['datafile'], data_dir)
        self.reset()


    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.load(df)
        return data


    def _setup_data(self, data_path, jsons_path):
        # loading directly from test file or val file or train file
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


    def _path(self, opt):
        # set up path to data (specific to each dataset)
        opt['datapath'] = "/tmp/parlai_data"
        data_dir = os.path.join(opt['datapath'], 'multiwoz21_trade',
'MULTIWOZ2.1')
        data_path = os.path.join(data_dir, 'data_trade_reformat.json')

        # build the data if it does not exist
        build(opt)

        return data_path, data_dir


    def num_examples(self):
        # each turn be seen as a individual dialog
        return len(self.messages)


    def num_episodes(self):
        return len(self.messages)


    def get(self, episode_idx, entry_idx=0):
        entry = self.messages[episode_idx]['context']
        episode_done = True
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx]['response']],
            'dial_id': self.messages[episode_idx]['dial_id'],
            'turn_num': self.messages[episode_idx]['turn_num'],
        }
        return action


class MultiWozTradeDSTTeacher(MultiWozTradeTeacher):
    """
    MultiWOZ2.1 Dialog State Tracking (DST) Teacher with TRADE preprocess,
    Each turn contains all the dialog history in the previous (turn["context"])
    and is expected to generate all previous dialog slots. Therefore, each turn
    is a individual episode.
    """
    MAX_TRAIN_DIALOGS = 5000
    CONTEXT_SWITCH_TEMPLATES = [" Anyways, getting back to the {}", " Coming back to the {}", " Alright, getting back to {}", "Getting back to the {}"]

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument(
            "--tasks_to_blend",
            type=str,
            default="",
        )
        return argparser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'multiwoz_trade'
        self.tasks_to_blend = []
        if opt['tasks_to_blend'] != "":
            self.tasks_to_blend = opt['tasks_to_blend'].split(',')
        self.opt = opt
        self.reset()
        self._init_other_tasks(opt)

    def _get_world(self, task_id):
        curr_task_opt = self.opt.copy()
        curr_task_opt['task'] = task_id
        curr_task_opt['datapath'] = os.path.join(curr_task_opt['datapath'],"blended_tasks")
        agent = RepeatLabelAgent(curr_task_opt)
        world = create_task(curr_task_opt, agent)
        return world

    def _get_dialog(self, world):
        turns = []  
        world.parley()
        turns.append(world.get_acts()[0])
        while not world.episode_done():
            world.parley()
            turns.append(world.get_acts()[0])
        return turns
    
    def _init_other_tasks(self, opt):
        self.dialogs_all_tasks = []
        for task_id in self.tasks_to_blend:
            world = self._get_world(task_id)
            self.dialogs_all_tasks.append([self._get_dialog(world) for _ in range(self.MAX_TRAIN_DIALOGS)])


    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type
slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        domains    = ["attraction", "hotel", "hospital", "restaurant",
"police", "taxi", "train"]
        slots_list = []

        # split according to ","
        str_split = slots_string.strip().split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in domains:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people",
"stay"]:
                    slot_type = slot[1]+" "+slot[2]
                    slot_val  = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val  = " ".join(slot[2:])
                if not slot_val == 'dontcare':
                    slots_list.append(domain+"--"+slot_type+"--"+slot_val)
        return slots_list


    def custom_evaluation(self, teacher_action: Message, labels, model_response: Message):
        """
        for dialog state tracking, we compute the joint goal accuracy, which is
        the percentage of the turns where the model correctly and precisely
        predicts all slots(domain, slot_type, slot_value).
        """
        resp = model_response.get('text')
        if not resp:
            return

        # extract ground truth from labels
        slots_truth = self._extract_slot_from_string(labels[0])

        # extract generated slots from model_response
        slots_pred = self._extract_slot_from_string(resp)

        self.metrics.add('joint goal acc', AverageMetric(set(slots_truth) == set(slots_pred)))

    def _can_interleave(self, turn):
        #sys_turn = turn["labels"][0]
        sys_turn = turn.split(" <system> ")[1]
        return "?" not in sys_turn   
    
    def _chunk_dialogs(self, dialogs, max_chunk_size=2):
        chunks = []
        chunk = []
        for turn in dialogs:
            if len(chunk) >= max_chunk_size and self._can_interleave(turn):
                chunks.append(chunk)
                chunk = []
            chunk.append(turn)
        chunks.append(chunk)
        return chunks

    def _blend_dialogs(self, multiwoz_turns, other_turn_chunks, domain):
        num_avail_positions_to_insert = len(multiwoz_turns)
        num_locs_to_insert = min(len(other_turn_chunks), num_avail_positions_to_insert)
        positions_to_insert = set(np.random.choice(num_avail_positions_to_insert, num_locs_to_insert,replace=False))
        blended_turns = []
        for turn_id, turn in enumerate(multiwoz_turns):
            if turn_id in positions_to_insert:
                for turn in other_turn_chunks.pop(0):
                    blended_turns.append(turn)
                if turn_id > 0:
                     context_switch_string = random.choice(self.CONTEXT_SWITCH_TEMPLATES).format(domain)
                     context_switch_turn = context_switch_string + "," + multiwoz_turns[turn_id]
                     blended_turns.append(context_switch_turn)
                else:
                    blended_turns.append(multiwoz_turns[turn_id])
            else:
                blended_turns.append(multiwoz_turns[turn_id])
        while other_turn_chunks:
            for turn in other_turn_chunks.pop(0):
                    blended_turns.append(turn)
            
        return blended_turns

    def _extend_dialog_history(self, self_entry, sys_response, domain):
        if not self.tasks_to_blend:
            return self_entry
        self_turns = self_entry.split(' <user> ')
        if len(self_turns) < 2:
            return self_entry
            
        dialogs_to_blend = random.choice(self.dialogs_all_tasks)
        dialog_to_blend = random.choice(dialogs_to_blend)
        turns = [f" {turn['text']} <system> {turn['labels'][0]}" for turn in dialog_to_blend]
        chunked_dialog = self._chunk_dialogs(turns)
        
        self_entry += ' <system> ' + sys_response
        self_turns = self_entry.split('<user>')
        if self_turns[0] == '':
            self_turns.pop(0)
        
        blended_turns = self._blend_dialogs(self_turns, chunked_dialog, domain)
        blended_context = "<user>" + " <user> ".join(blended_turns)
        return blended_context
        
    def get(self, episode_idx, entry_idx=0):
        entry = self.messages[episode_idx]['context'] # includes all previous dialog history
        episode_done = True     # each turn is a individual episode
        domain = self.messages[episode_idx]['domain']
        action = {
            'id': self.id,
            'text': self._extend_dialog_history(entry, self.messages[episode_idx]['response'], domain),
            'domain':domain,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx]['slots']],
            'dial_id': self.messages[episode_idx]['dial_id'],
            'turn_num': self.messages[episode_idx]['turn_num'],
        }
        return action


class DefaultTeacher(MultiWozTradeDSTTeacher):
    pass
