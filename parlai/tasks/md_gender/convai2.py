#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.convai2.agents import BothTeacher as OrigConvai2Teacher
from parlai.tasks.md_gender.build import build
from parlai.utils.misc import warn_once

import parlai.tasks.md_gender.utils as gend_utils

from copy import deepcopy
import json
import random
import os


gender_convert = {'man': 'male', 'woman': 'female', 'neutral': 'unknown', None: None}


def convert_text(text):
    """
    Normalize text.
    """
    new_text = text.lower()
    for x in gend_utils.CONTRACTION_SPACES:
        if x[1] in text:
            new_text = new_text.replace(x[1], x[0])
    for x in gend_utils.CONTRACTION_LEFT_SPACES:
        if x[1] in text:
            new_text = new_text.replace(x[1], x[0])
    for x in gend_utils.CONTRACTION_RIGHT_SPACES:
        if x[1] in text:
            new_text = new_text.replace(x[1], x[0])

    return new_text


class Convai2Teacher(FixedDialogTeacher):
    """
    Predict the gender of character given the dialogue utterance.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        agent = argparser.add_argument_group('ConvAI2 gender')
        agent.add_argument(
            '--convai2-use-probably',
            type='bool',
            default=True,
            help='Use the probable gender of the persona as '
            'determined by crowdworkers, instead of the '
            'definitive one',
        )
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
        self.label_candidates = [gend_utils.MASC, gend_utils.FEM, gend_utils.NEUTRAL]
        self.use_probably = opt['convai2_use_probably']
        self.labels_to_use = opt['labels_to_use']
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train

        if shared and 'data' in shared:
            self.data = shared['data']
            self.persona_map = shared['persona_map']
        else:
            self.missing_cnt = 0
            self._load_persona_map(opt)
            self._setup_data(opt)
            if (self.is_train and opt['balance']) or (
                self.is_valid and opt['balance_valid']
            ):
                exclude_lst = gend_utils.ABOUT_CANDS
                self.data = gend_utils.balance_data(
                    self.data, exclude_labels=exclude_lst
                )

        self.label_candidates = gend_utils.ALL_CANDS

        opt = deepcopy(opt)
        super().__init__(opt, shared)
        self.reset()

    def _load_persona_map(self, opt):
        build(opt)
        persona_map_path = os.path.join(
            opt['datapath'],
            'md_gender',
            'data_to_release',
            'convai2',
            'convai2_all_personas_map.json',
        )
        with open(persona_map_path, 'rb') as f:
            self.persona_map = json.load(f)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def get_gender_annotations(self, lst):
        """
        Take a list of personas and find the gender.
        """
        new_lst = sorted([convert_text(persona) for persona in lst])
        persona_key = ' '.join(new_lst)
        if persona_key in self.persona_map:
            return self.persona_map[persona_key]
        else:
            return {'gender': 'neutral', 'probably': None, 'missing': True}

    def get_genders(self, your_persona, partner_persona):
        partner_persona_gender = self.get_gender_annotations(partner_persona)
        your_persona_gender = self.get_gender_annotations(your_persona)
        if 'missing' in partner_persona_gender:
            self.missing_cnt += 1
        if 'missing' in your_persona_gender:
            self.missing_cnt += 1

        partner_prob = None
        your_prob = None
        if self.use_probably:
            partner = partner_persona_gender['probably']
            if partner is None:
                partner = partner_persona_gender['gender']
            your = your_persona_gender['probably']
            if your is None:
                your = your_persona_gender['gender']
        else:
            partner = partner_persona_gender['gender']
            your = your_persona_gender['gender']
            partner_prob = partner_persona_gender['probably']
            your_prob = your_persona_gender['probably']

        your = gender_convert[your]
        your_prob = gender_convert[your_prob]
        partner = gender_convert[partner]
        partner_prob = gender_convert[partner_prob]

        return your, your_prob, partner, partner_prob

    def _setup_data(self, opt):
        counts = {
            'partner': {gend_utils.UNKNOWN: 0, gend_utils.FEM: 0, gend_utils.MASC: 0},
            'self': {gend_utils.UNKNOWN: 0, gend_utils.FEM: 0, gend_utils.MASC: 0},
        }

        dt = opt['datatype'].split(':')[0]
        if dt == 'test':
            warn_once('No test set; switching to valid')
            dt = 'valid'

        # build data
        print('[ Building data ... ]')
        new_eps = []
        orig_teacher = OrigConvai2Teacher(opt)
        total_exs = orig_teacher.num_examples()
        num_exs = 0
        while num_exs < total_exs:
            current_episode = []
            episode_done = False

            while not episode_done:
                # TODO: eventually all teachers should return Messages, so
                # we should assert this
                action = Message(orig_teacher.act())
                current_episode.append(action)
                episode_done = action.get('episode_done', False)
                num_exs += 1

            # now we have the entire episode,... do something
            first_ex = current_episode[0]
            first_ex_text = []
            partner_persona = []
            your_persona = []
            for line in first_ex['text'].split('\n'):
                # NOTE: we flip "your" and "partner" here since we are taking the 'text'
                # field instead of the 'label'
                if 'partner\'s persona: ' in line:
                    your_persona.append(line.split('partner\'s persona: ')[1])
                elif 'your persona: ' in line:
                    partner_persona.append(line.split('your persona: ')[1])
                else:
                    first_ex_text.append(line)

            your, your_prob, partner, partner_prob = self.get_genders(
                your_persona, partner_persona
            )

            for i, ex in enumerate(current_episode):
                counts['self'][your] += 1
                counts['partner'][partner] += 1
                if i == 0:
                    text = '\n'.join(first_ex_text)
                else:
                    text = ex['text']
                new_ex = {
                    'text': text,
                    'episode_done': True,
                    'your_persona': '\n'.join(your_persona),
                    'partner_persona': '\n'.join(partner_persona),
                    'id': 'ConvAI2 Gender',
                }
                if not self.use_probably:
                    new_ex['partner_prob'] = partner_prob
                    new_ex['your_prob'] = your_prob

                if your is not None and self.labels_to_use != 'partner':
                    # Get the your task
                    labels = [f'SELF:{your}']
                    your_ex = deepcopy(new_ex)
                    your_ex['labels'] = labels
                    your_ex['class_type'] = 'self'
                    new_eps.append(your_ex)

                if partner is not None and self.labels_to_use != 'self':
                    # Get the partner task
                    labels = [f'PARTNER:{partner}']
                    partner_ex = deepcopy(new_ex)
                    partner_ex['labels'] = labels
                    partner_ex['class_type'] = 'partner'
                    new_eps.append(partner_ex)

        if self.labels_to_use == 'all' and self.add_unknown_classes:
            # load about data
            all_about_data = gend_utils.get_inferred_about_data(
                self.opt['task'], self.opt
            )
            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                to_samp = int(sample_rate * len(all_about_data))
                sampled = random.sample(all_about_data, to_samp)
                new_eps += sampled
            else:
                new_eps += all_about_data

        if self.is_train:
            random.shuffle(new_eps)

        self.data = new_eps
        print(f'Missing cnt: {self.missing_cnt} / {len(self.data) * 2}')
        for x in ['self', 'partner']:
            print(f'Totals for {x}:')
            subtot = sum(counts[x].values())
            for k, v in counts[x].items():
                print(f'\t{k}: {v} ({v / subtot})')

    def get(self, episode_idx, entry_idx):
        ep = self.data[episode_idx]
        # add label candidates
        # we do not want to hardcode these in the saved data, in case
        # we try to change them
        class_type = ep['class_type']

        if gend_utils.UNKNOWN in ep['labels'][0]:
            # for unknown, we alternate M/F during training
            ep['labels'] = gend_utils.UNKNOWN_LABELS[class_type]

        ep['label_candidates'] = self.label_candidates[class_type]
        return Message(ep)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['persona_map'] = self.persona_map
        return shared
