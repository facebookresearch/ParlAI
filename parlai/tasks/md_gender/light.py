#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.md_gender.build import build
import parlai.tasks.md_gender.utils as gend_utils

from copy import deepcopy
import json
import pickle
import os
import random

PERSONA_PATH = 'md_gender/data_to_release/light/personas.json'
LIGHT_DATA_PATH = 'md_gender/data_to_release/light/data/{}_convs.pkl'


class LIGHTTeacher(FixedDialogTeacher):
    """
    Predict the gender of character given the dialogue utterance.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        agent = argparser.add_argument_group('LIGHT gender')
        agent.add_argument(
            '--labels-to-use',
            type=str,
            default='all',
            choices=['all', 'self', 'partner'],
            help='Which labels to use for this teacher',
        )

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.fixed_random = random.Random(42)
        self.label_candidates = gend_utils.ALL_CANDS
        self.labels_to_use = opt['labels_to_use']
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self._setup_data(opt)
            if (self.is_train and opt['balance']) or (
                self.is_valid and opt['balance_valid']
            ):
                to_exclude = gend_utils.ABOUT_CANDS
                self.data = gend_utils.balance_data(
                    self.data, exclude_labels=to_exclude
                )

        opt = deepcopy(opt)
        super().__init__(opt, shared)
        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def _setup_data(self, opt):
        build(opt)
        datatype = opt['datatype']
        dt = datatype.split(':')[0]
        # Build a map from persona to gender
        persona_map = {}
        personas = json.load(open(os.path.join(opt['datapath'], PERSONA_PATH), 'rb'))[
            'old'
        ]
        for gender, lst in personas.items():
            for x in lst:
                persona_map[int(x['char_id'])] = {'name': x['name'], 'gender': gender}

        # Build a list of dialogue utterances and associated persona IDs
        light_world = pickle.load(
            open(os.path.join(opt['datapath'], LIGHT_DATA_PATH.format(dt)), 'rb')
        )
        utt_to_pers = []
        for x in light_world:
            for act in x['conv_info']['acts']:
                text = act['text']
                p_uid = act['id'].lower()
                self_char_id = None
                partner_char_id = None
                for y in x['conv_info']['characters']:
                    # if identifying self utterances, grab your own id
                    if y[0].lower() == p_uid:
                        self_char_id = y[1]['id']
                        self_name = y[0].lower()
                    # else grab the partner's id
                    elif y[0].lower() != p_uid:
                        partner_char_id = y[1]['id']
                        partner_name = y[0].lower()
                if self_char_id is not None and partner_char_id is not None:
                    utt_to_pers.append(
                        {
                            'text': text,
                            'self_id': self_char_id,
                            'self_name': self_name,
                            'partner_id': partner_char_id,
                            'partner_name': partner_name,
                        }
                    )

        self.data = []
        missing = 0

        counts = {
            'partner': {gend_utils.UNKNOWN: 0, gend_utils.FEM: 0, gend_utils.MASC: 0},
            'self': {gend_utils.UNKNOWN: 0, gend_utils.FEM: 0, gend_utils.MASC: 0},
        }

        for x in utt_to_pers:
            if x['self_id'] in persona_map and x['partner_id'] in persona_map:
                self_gender = persona_map[x['self_id']]['gender']
                partner_gender = persona_map[x['partner_id']]['gender']
                act = {
                    'text': x['text'].lower(),
                    'self_id': x['self_id'],
                    'partner_id': x['partner_id'],
                    'id': 'LIGHT Gender',
                    'episode_done': True,
                }
                if self_gender == gend_utils.NEUTRAL:
                    # not True neutral
                    self_gender = gend_utils.UNKNOWN
                if partner_gender == gend_utils.NEUTRAL:
                    # not True neutral
                    partner_gender = gend_utils.UNKNOWN
                if self_gender is not None and self.labels_to_use != 'partner':
                    labels = [f'SELF:{self_gender}']
                    self_act = deepcopy(act)
                    self_act['labels'] = labels
                    self_act['class_type'] = 'self'
                    self.data.append(self_act)
                if partner_gender is not None and self.labels_to_use != 'self':
                    labels = [f'PARTNER:{partner_gender}']
                    partner_act = deepcopy(act)
                    partner_act['labels'] = labels
                    partner_act['class_type'] = 'partner'
                    self.data.append(partner_act)

                counts['partner'][partner_gender] += 1
                counts['self'][self_gender] += 1
            else:
                missing += 1

        if self.labels_to_use == 'all' and self.add_unknown_classes:
            # load about data
            all_about_data = gend_utils.get_inferred_about_data(
                self.opt['task'], self.opt
            )
            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                # do something here
                to_samp = int(sample_rate * len(all_about_data))
                sampled = random.sample(all_about_data, to_samp)
                self.data += sampled
            else:
                self.data += all_about_data

        total = len(self.data)
        print(f'Total: {total}')
        for x in ['self', 'partner']:
            print(f'Totals for {x}:')
            subtot = sum(counts[x].values())
            for k, v in counts[x].items():
                print(f'\t{k}: {v} ({v / subtot})')

    def get(self, episode_idx, entry_idx):
        ep = self.data[episode_idx]
        class_type = ep['class_type']
        if gend_utils.UNKNOWN in ep['labels'][0]:
            # during training, we flip between labels
            ep['labels'] = gend_utils.UNKNOWN_LABELS[class_type]
        ep['label_candidates'] = self.label_candidates[class_type]
        return Message(ep)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared
