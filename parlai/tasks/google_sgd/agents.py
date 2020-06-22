#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Google The Schema-Guided Dialogue(SGD) Dataset implementation for ParlAI.
"""

import os
import copy
import json
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
import parlai.utils.logging as logging

import parlai.tasks.google_sgd.build as build_
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task, create_task_world

def _load_secondary_world(opt):
    if opt['stask'] == 'None':
        return None
    s_opt = copy.deepcopy(opt)
    s_opt['task'] = s_opt['stask']
    s_opt['datatype'] = 'train'
    s_agent = RepeatLabelAgent(s_opt)
    s_world = create_task_world(s_opt, [s_agent])
    return s_world


class Text2API2TextTeacher(DialogTeacher):
    """
    Abstract data loader.
    """
    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument('-st', '--stask', type=str, default='None')
        return argparser

    def __init__(self, opt: Opt, shared=None):
        self.fold = opt['datatype'].split(':')[0]
        opt['datafile'] = self.fold
        self.dpath = os.path.join(opt['datapath'], 'google_sgd')
        if shared is None:
            warn_once(
                "Google SGD is a beta dataset, and format may significantly change."
            )
            build_.build(opt)
        self._s_world = _load_secondary_world(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold):
        dataset_fold = 'dev' if fold == 'valid' else fold
        fold_path = os.path.join(self.dpath, dataset_fold)
        schema_file = os.path.join(fold_path, 'schema.json')
        with open(schema_file, 'r') as f:
            schema_lookup = {}
            for schema in json.load(f):
                schema_lookup[schema['service_name']] = schema

        dialogs = []
        for file_id in range(1, build_.fold_size(dataset_fold) + 1):
            filename = os.path.join(fold_path, f'dialogues_{file_id:03d}.json')
            with open(filename, 'r') as f:
                dialogs += json.load(f)
        return schema_lookup, dialogs

    def _get_api_call_and_results(self, sys_turn, schema_lookup):
        api_call = {}
        api_resp = {}
        for frame in sys_turn['frames']:
            if 'service_call' in frame:
                # API CALL
                method = frame['service_call']['method']
                for slot_type, slot_value in frame['service_call'][
                    'parameters'
                ].items():
                    api_call[f'{method}.{slot_type}'] = slot_value
                assert 'service_results' in frame

            # API Resp
            if 'actions' in frame:
                for action in frame['actions']:
                    slot_type = action['slot']
                    slot_value = action['canonical_values']
                    api_resp[slot_type] = slot_value
        return api_call, api_resp

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get('text')
        if not resp:
            return

        if teacher_action['type'] == 'apicall' and resp.startswith('apicall: '):
            gold = teacher_action['slots']
            slot_strs = resp[9:].split(' ; ')
            parsed = {}
            for slot_str in slot_strs:
                if ' = ' not in slot_str:
                    continue
                try:
                    name, value = slot_str.split(' = ')
                    parsed[name] = value
                except:
                    continue

            # slot precision
            for k, v in parsed.items():
                self.metrics.add('slot_p', AverageMetric(v == gold.get(k)))
            # slot recall
            for k, v in gold.items():
                self.metrics.add('slot_r', AverageMetric(v == parsed.get(k)))
        elif teacher_action['type'] == 'apiresp':
            delex_resp = self._delex(resp, teacher_action['slots'])
            delex_label = self._delex(labels[0], teacher_action['slots'])
            self.metrics.add(
                'delex_bleu', BleuMetric.compute(delex_resp, [delex_label])
            )

    def _delex(self, text, slots):
        delex = text
        for slot, values in slots.items():
            assert isinstance(values, list)
            for value in values:
                delex = delex.replace(value, slot)
        return delex

    def _api_dict_to_str(self, apidict):
        return ' ; '.join(f'{k} = {v}' for k, v in apidict.items())

    def _get_secondary_acts(self, max_secondary_turns=2):
        secondary_acts = []
        for _ in range(max_secondary_turns): 
            self._s_world.parley()
            acts = self._s_world.get_acts()[0]
            input_text = acts['text']
            output_text = acts['labels'][0]
            secondary_act = {
            "text" : input_text,
            "labels": output_text,
            "type": "text",
                }
            secondary_acts.append(secondary_act)
        return secondary_acts


    def setup_data(self, fold):
        max_primary_turns = 2
        local_num_primary_turns = 0
        for primary_act, is_first_turn in self._setup_data(fold):
            if primary_act['type'] == 'apiresp' and \
                local_num_primary_turns >= max_primary_turns and \
                self._s_world is not None:
                secondary_act_chunks = self._get_secondary_acts()
                while secondary_act_chunks:
                    secondary_act = secondary_act_chunks.pop(0)
                    yield secondary_act, False
                local_num_primary_turns = 0
            else:
                local_num_primary_turns += 1
                yield primary_act, is_first_turn

            
    def _setup_data(self, fold):
        schema_lookup, dialogs = self._load_data(fold)
        for dialog in dialogs:
            # services = dialog['services']
            turns = dialog['turns']
            num_turns = len(turns)
            for turn_id in range(0, num_turns, 2):
                is_first_turn = turn_id == 0

                user_turn = turns[turn_id]
                sys_turn = turns[turn_id + 1]
                api_call, api_results = self._get_api_call_and_results(
                    sys_turn, schema_lookup
                )
                call_str = self._api_dict_to_str(api_call)
                resp_str = self._api_dict_to_str(api_results)
                if not api_call and not api_results:
                    # input: user_turn, output: sys_turn
                    yield {
                        'text': user_turn['utterance'],
                        'label': sys_turn['utterance'],
                        'type': 'text',
                    }, is_first_turn
                elif not api_call and api_results:
                    yield {
                        'text': f"{user_turn['utterance']} api_resp: {resp_str}",
                        'label': sys_turn['utterance'],
                        'type': 'apiresp',
                        'slots': api_results,
                    }, is_first_turn
                elif api_call and api_results:
                    # input: user_turn, output: api_call
                    yield {
                        'text': user_turn['utterance'],
                        'label': f'apicall: {call_str}',
                        'type': 'apicall',
                        'slots': api_call,
                    }, is_first_turn

                    # system turn, input : api results, output : assistant turn
                    yield {
                        'text': f"api_resp: {resp_str}",
                        'label': sys_turn['utterance'],
                        'type': 'apiresp',
                        'slots': api_results,
                    }, False
                else:
                    assert False, "API call without API results! Check Dataset!"


class Text2TextTeacher(Text2API2TextTeacher):
    def setup_data(self, fold):
        schema_lookup, dialogs = self._load_data(fold)
        for dialog in dialogs:
            turns = dialog['turns']
            num_turns = len(turns)
            for turn_id in range(0, num_turns, 2):
                if turn_id == 0:
                    is_first_turn = True
                else:
                    is_first_turn = False

                user_turn = turns[turn_id]
                sys_turn = turns[turn_id + 1]
                # input: user_turn, output: sys_turn
                yield {
                    'text': user_turn['utterance'],
                    'label': sys_turn['utterance'],
                    'type': 'text',
                }, is_first_turn


class DefaultTeacher(Text2API2TextTeacher):
    pass
