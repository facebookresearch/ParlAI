#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.md_gender.build import build
import parlai.tasks.md_gender.utils as gend_utils

import json
import os
import random

NEW_DATAFILE = 'md_gender/data_to_release/new_data/data.jsonl'


class MdGenderTeacher(FixedDialogTeacher):
    """
    MDGender data collected on Mechanical Turk.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        agent = argparser.add_argument_group('New data gender')
        agent.add_argument(
            '--labels-to-use',
            type=str,
            default='all',
            choices=['all', 'self', 'partner', 'about'],
            help='Which labels to use for this teacher',
        )
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.labels_to_use = opt['labels_to_use']
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.label_candidates = gend_utils.ALL_CANDS

        if shared is None:
            # set map
            self.data = self._setup_data(opt)
            if (self.is_train and opt['balance']) or (
                self.is_valid and opt['balance_valid']
            ):
                to_exclude = (
                    ['ABOUT:non-binary'],
                )  # not enough non-binary examples to balance
                self.data = gend_utils.balance_data(
                    self.data, key='labels', exclude_labels=to_exclude
                )
        else:
            self.data = shared['data']
        super().__init__(opt, shared)
        self.reset()

    def _get_convos(self, opt):
        """
        Get the test/train/valid split.
        """
        build(opt)
        with open(os.path.join(opt['datapath'], NEW_DATAFILE), 'r') as f:
            ex_jsons = f.read().splitlines()

        convos = [json.loads(ex) for ex in ex_jsons]

        return convos

    def _setup_data(self, opt):
        # Load map from image ID to gender
        # TODO: proper train/test/valid splits
        data = []
        convos = self._get_convos(opt)
        for convo in convos:
            class_type = convo['class_type']
            if self.labels_to_use in ['all', class_type]:
                data.append(convo)

        if self.is_train:
            random.shuffle(data)

        gend_utils.get_data_stats(data, key='labels')

        return data

    def get(self, episode_idx, entry_idx=0):
        ex = self.data[episode_idx]
        class_type = ex['class_type']
        ex['label_candidates'] = gend_utils.ALL_CANDS[class_type]
        if 'unknown' in ex['labels'][0]:
            ex['labels'] = gend_utils.UNKNOWN_LABELS[class_type]
        return Message(ex)

    def num_examples(self):
        return len(self.data)

    def num_episodes(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared
