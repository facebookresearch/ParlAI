#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Google The Schema-Guided Dialogue(SGD) Dataset implementation for ParlAI.
"""

import os
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once
import json

import parlai.tasks.google_sgd.build as build_


class Text2API2TextTeacher(DialogTeacher):
    """
    Abstract data loader.
    """

    def __init__(self, opt: Opt, shared=None):
        self.fold = opt['datatype'].split(':')[0]
        opt['datafile'] = self.fold
        self.dpath = os.path.join(opt['datapath'], 'google_sgd')
        if shared is None:
            warn_once(
                "Google SGD is a beta dataset, and format may significantly change."
            )
            build_.build(opt)
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
        api_call = ''
        api_resp = ''
        for frame in sys_turn['frames']:
            if 'service_call' in frame:
                # API CALL
                method = frame['service_call']['method']
                for slot_type, slot_value in frame['service_call'][
                    'parameters'
                ].items():
                    api_call += f"{method}.{slot_type} = {slot_value} ;"
                assert 'service_results' in frame

                # API Resp
                # for action in frame['actions']:
                #    slot_type = action['slot']
                #    slot_value = action['canonical_values']
                #    api_resp += f"{method}.{slot_type} = {slot_value} ;"
                for result in frame['service_results']:
                    for slot_type, slot_value in result.items():
                        api_resp += f"{method}.{slot_type} = {slot_value} ;"
        return api_call, api_resp

    def setup_data(self, fold):
        schema_lookup, dialogs = self._load_data(fold)
        for dialog in dialogs:
            # services = dialog['services']
            turns = dialog['turns']
            num_turns = len(turns)
            for turn_id in range(0, num_turns, 2):
                if turn_id == 0:
                    is_first_turn = True
                else:
                    is_first_turn = False

                user_turn = turns[turn_id]
                sys_turn = turns[turn_id + 1]
                api_call, api_results = self._get_api_call_and_results(
                    sys_turn, schema_lookup
                )
                if not api_call:
                    # input: user_turn, output: sys_turn
                    yield {
                        'text': user_turn['utterance'],
                        'label': sys_turn['utterance'],
                        'type': 'text',
                    }, is_first_turn
                else:
                    # input: user_turn, output: api_call
                    yield {
                        'text': user_turn['utterance'],
                        'label': api_call,
                        'type': 'apicall',
                    }, is_first_turn

                    # system turn, input : api results, output : assistant turn
                    yield {
                        'text': api_results,
                        'label': sys_turn['utterance'],
                        'type': 'apiresp',
                    }, False


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
