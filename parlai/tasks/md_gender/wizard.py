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
    UNKNOWN,
    balance_data,
    add_common_args,
    ALL_CANDS,
    SELF_CANDS,
    PARTNER_CANDS,
    UNKNOWN_LABELS,
    get_data_stats,
)


def get_gender_data(datatype, binary=False):
    """
    Load data from the checkpoint
    """
    dt = datatype.split(':')[0]
    fle = f'/checkpoint/parlai/tasks/gender_multiclass/wizard/{dt}.jsonl'

    data = []
    with open(fle, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            ex = json.loads(line)
            ex['class_type'] = 'about'
            gender = ex['gender']
            if binary and gender in [NEUTRAL, NONBINARY]:
                # binary classification, skip these examples
                continue
            ex['label'] = f'ABOUT:{gender}'
            data.append(ex)

    return data


class WizardTeacher(FixedDialogTeacher):
    """
    Wizard of Wikipedia Gender gender.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train
        self.label_candidates = ALL_CANDS

        if shared is None:
            # set map
            self.data = self._setup_data(opt)
            if (self.is_train and opt['balance']) or (
                self.is_valid and opt['balance_valid']
            ):
                # don't want to balance the unknown data
                to_exclude = [f'SELF:{UNKNOWN}', f'PARTNER:{UNKNOWN}']
                self.data = balance_data(
                    self.data, key='label', exclude_labels=to_exclude
                )
        else:
            self.data = shared['data']
        super().__init__(opt, shared)
        self.reset()

    def _setup_data(self, opt):
        # Load map from image ID to gender
        data = get_gender_data(opt['datatype'], binary=opt['binary'])

        extra_data = []
        if self.add_unknown_classes:
            for ex in data:
                self_ex = deepcopy(ex)
                self_ex['label'] = f'SELF:{UNKNOWN}'
                self_ex['class_type'] = 'self'
                extra_data.append(self_ex)

                partner_ex = deepcopy(ex)
                partner_ex['label'] = f'PARTNER:{UNKNOWN}'
                partner_ex['class_type'] = 'partner'
                extra_data.append(partner_ex)

        if len(extra_data) > 0:
            # possibly sample unknown classes
            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                to_samp = int(sample_rate * len(extra_data))
                sampled = random.sample(extra_data, to_samp)
                data += sampled
            else:
                data += extra_data

        if self.is_train:
            random.shuffle(data)

        get_data_stats(data, key='label', lst=False)

        return data

    def get(self, episode_idx, entry_idx=0):
        ex = self.data[episode_idx]
        text = ex['text']
        class_type = ex['class_type']
        if class_type in ['self', 'partner']:
            labels = UNKNOWN_LABELS[class_type]
        else:
            labels = [ex['label']]
        return Message(
            {
                'text': text,
                'labels': labels,
                'episode_done': True,
                'label_candidates': self.label_candidates[class_type],
                'id': 'Wizard Gender',
                'title': ex['chosen_topic'],
            }
        )

    def num_examples(self):
        return len(self.data)

    def num_episodes(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared
