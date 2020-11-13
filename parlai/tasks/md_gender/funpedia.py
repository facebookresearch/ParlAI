#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
import parlai.tasks.md_gender.utils as gend_utils
from parlai.tasks.md_gender.build import build

from copy import deepcopy
import json
import os
import random


class FunpediaTeacher(FixedDialogTeacher):
    """
    Funpedia Gender gender.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train
        self.label_candidates = gend_utils.ALL_CANDS

        if shared is None:
            # set map
            self.data = self._setup_data(opt)
            if (self.is_train and opt['balance']) or (
                self.is_valid and opt['balance_valid']
            ):
                to_exclude = gend_utils.PARTNER_CANDS + gend_utils.SELF_CANDS
                self.data = gend_utils.balance_data(
                    self.data, key='labels', exclude_labels=to_exclude
                )
        else:
            self.data = shared['data']
        super().__init__(opt, shared)
        self.reset()

    def _load_gender_data(self, opt):
        build(opt)
        dt = opt['datatype'].split(':')[0]
        fle = os.path.join(
            opt['datapath'], 'md_gender', 'data_to_release', 'funpedia', f'{dt}.jsonl'
        )

        data = []
        with open(fle, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                ex = json.loads(line)
                gender = ex['gender']
                ex['labels'] = [f'ABOUT:{gender}']
                ex['class_type'] = 'about'
                data.append(ex)

        return data

    def _setup_data(self, opt):
        # Load map from image ID to gender
        data = self._load_gender_data(opt)

        # Possibly add extra examples
        extra_data = []
        if self.add_unknown_classes:
            for ex in data:
                # add self examples
                self_ex = deepcopy(ex)
                self_ex['class_type'] = 'self'
                # not True neutral, so we flip between
                self_ex['labels'] = gend_utils.UNKNOWN_LABELS['self']
                extra_data.append(self_ex)
                # add partner examples
                partner_ex = deepcopy(ex)
                partner_ex['labels'] = gend_utils.UNKNOWN_LABELS['partner']
                partner_ex['class_type'] = 'partner'
                extra_data.append(partner_ex)

            # now sample the data
            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                to_samp = int(sample_rate * len(extra_data))
                sampled = random.sample(extra_data, to_samp)
                data += sampled
            else:
                data += extra_data

        if self.is_train:
            random.shuffle(data)

        gend_utils.get_data_stats(data, key='labels')

        return data

    def get(self, episode_idx, entry_idx=0):
        ex = self.data[episode_idx]
        class_type = ex['class_type']
        return Message(
            {
                'text': ex['text'],
                'labels': ex['labels'],
                'episode_done': True,
                'label_candidates': self.label_candidates[class_type],
                'id': 'Funpedia Gender',
                'title': ex['title'],
                'persona': ex['persona'],
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
