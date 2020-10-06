#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher

from copy import deepcopy
import json
import os
import random
from tqdm import tqdm
from typing import List, Tuple

from parlai.tasks.md_gender.utils import (
    MASC,
    FEM,
    NEUTRAL,
    NONBINARY,
    add_common_args,
    ALL_CANDS,
    EMPTY_LABELS,
    get_inferred_about_data,
)


def get_gender_data(datatype):
    """
    Load data from the checkpoint
    """
    dt = datatype.split(':')[0]
    data = []
    folder = '/checkpoint/parlai/tasks/gender_multiclass/yelp'
    if dt == 'train':
        gender_cnt = {MASC: 0, FEM: 0}
        fle = os.path.join(folder, 'classtrain.txt')
        with open(fle, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                gender, text = line.split('\t')
                data.append(
                    {'text': text, 'labels': [f'SELF:{gender}'], 'class_type': 'self'}
                )
                gender_cnt[gender] += 1
        print('Trainset gender cnts:\n' + '=' * 50)
        tot = sum(gender_cnt.values())
        for k, v in gender_cnt.items():
            print(f'{k}: {v} ({v / tot})')
        print(f'TOTAL: {tot}')
    else:
        if dt == 'valid':
            dt = 'dev'
        # female data
        female_fle = os.path.join(folder, f'female_only.{dt}.en')
        # male data
        male_fle = os.path.join(folder, f'male_only.{dt}.en')
        with open(female_fle, 'r') as f:
            with open(male_fle, 'r') as m:
                f_lines = f.read().splitlines()
                m_lines = m.read().splitlines()
            for f_line, m_line in zip(f_lines, m_lines):
                # alternate this
                data.append(
                    {'text': f_line, 'labels': [f'SELF:{FEM}'], 'class_type': 'self'}
                )
                data.append(
                    {'text': m_line, 'labels': [f'SELF:{MASC}'], 'class_type': 'self'}
                )

    return data


class YelpTeacher(FixedDialogTeacher):
    """
    Image Chat gender.

    TODO: there are UNKs in here, need to consider how to handle them...
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train
        self.label_candidates = ALL_CANDS

        if shared is None:
            # set map
            self.data = self._setup_data(opt)
        else:
            self.data = shared['data']
        super().__init__(opt, shared)
        self.reset()

    def _setup_data(self, opt):
        # Load map from image ID to gender
        data = get_gender_data(opt['datatype'])

        extra_data = []
        if self.add_unknown_classes:
            # load about data (unknown but inferred)
            extra_data = get_inferred_about_data(self.opt['task'], self.opt['datatype'])

            # now create partner/TO data: true neutral
            for ex in data:
                partner_ex = deepcopy(ex)
                partner_ex['labels'] = [f'PARTNER:{NEUTRAL}']
                partner_ex['class_type'] = 'neutral'
                extra_data.append(ex)

            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                to_samp = int(sample_rate * len(extra_data))
                sampled = random.sample(extra_data, to_samp)
                data += sampled
            else:
                data += extra_data

        data = data + extra_data
        if self.is_train:
            random.shuffle(data)

        return data

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep['label_candidates'] = self.label_candidates[ep['class_type']]
        ep['id'] = 'Yelp Gender'
        ep['episode_done'] = True
        return Message(ep)

    def num_examples(self):
        return len(self.data)

    def num_episodes(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared
