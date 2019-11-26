#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Dialogue safety related datasets and teachers.
"""

import parlai.core.build_data as build_data
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher

from .base_agent import _BaseSafetyTeacher
from .build import build

import copy
import json
import os
import random
import sys as _sys


# Constants
OK_CLASS = '__ok__'
NOT_OK_CLASS = '__notok__'
MULTI_TURN_DATA = 'multi_turn_safety.json'


class StandardTeacher(_BaseSafetyTeacher):
    """
    Data from the standard collection described in the paper `Build it Break it Fix it
    for Dialogue Safety: Robustness from Adversarial Human Attack`
    (<https://arxiv.org/abs/1908.06083>)

    To see data from rounds 1, 2, and 3, try running:
    `python examples/display_data.py -t dialogue_safety:standard --round 3`

    To see data from round 2 only, try running:
    `python examples/display_data.py -t dialogue_safety:standard --round 2
     --round-only True`
    """

    def _load_data_dump(self):
        with open(self.data_path, 'rb') as f:
            dump = json.load(f)
        return dump['standard']


class AdversarialTeacher(_BaseSafetyTeacher):
    """
    Data from the adversarial collection described in the paper `Build it Break it Fix
    it for Dialogue Safety: Robustness from Adversarial Human Attack`
    (<https://arxiv.org/abs/1908.06083>)

    To see data from rounds 1, 2, and 3, try running:
    `python examples/display_data.py -t dialogue_safety:adversarial --round 3`

    To see data from round 2 only, try running:
    `python examples/display_data.py -t dialogue_safety:adversarial --round 2
     --round-only True`
    """

    def _load_data_dump(self):
        with open(self.data_path, 'rb') as f:
            dump = json.load(f)
        return dump['adversarial']


class MultiturnTeacher(FixedDialogTeacher):
    """
    Data from the multi-turn adversarial collection described in the paper `Build it
    Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack`
    (<https://arxiv.org/abs/1908.06083>)

    To see data containing multi-turn conversations, try running
    `python examples/display_data.py -t dialogue_safety:multiturn`.

    Run the above command with the flag `--single-turn True` to only see the
    single turn data.
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Multiturn Safety Teacher Args')
        parser.add_argument(
            '--single-turn',
            type='bool',
            default=False,
            help='only include the single turn data and not the context info',
        )

    def __init__(self, opt, shared=None):
        build(opt['datapath'])  # download the data
        self.opt = opt
        self.data_path = os.path.join(
            opt['datapath'], 'dialogue_safety', MULTI_TURN_DATA
        )

        self.fixed_random = random.Random(42)
        self.single_turn = opt['single_turn']

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self._setup_data(opt['datatype'])

        super().__init__(opt, shared)
        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

    def _setup_data(self, datatype):
        dt = datatype.split(':')[0]
        self.all_data = json.load(open(self.data_path, 'rb'))
        data = self.all_data[dt]
        if self.single_turn:
            # remove
            new_data = []
            for datum in data:
                datum['text'] = datum['text'].split('\n')[-1]
                new_data.append(datum)
            self.data = new_data
        else:
            self.data = data

    def get(self, episode_idx, entry_idx):
        return Message(self.data[episode_idx])


class WikiToxicCommentsTeacher(FixedDialogTeacher):
    """
    Dataset of comments from Wikipedia's Talk page edits. Taken from the Toxic Comments
    Classification Challenge on Kaggle.

    <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data>

    We convert this data to a binary classification task.
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Kaggle Toxic Comment Classification Data')
        parser.add_argument(
            '--use-test-set',
            type='bool',
            default=False,
            help='Defaults to 80/10/10 train/test/valid split of train set. '
            'Else, will partition train data into 90/10 train/valid and '
            'use the test set as is.',
        )
        parser.add_argument(
            '--balance-data',
            type='bool',
            default=False,
            help='Balances the data so there are equal numbers of OK and NOT '
            'OK training data',
        )
        return parser

    def __init__(self, opt, shared=None):
        self.fixed_random = random.Random(42)

        self.use_test_set = opt['use_test_set']
        self.balance_data = opt['balance_data']

        self.data_path = os.path.join(
            opt['datapath'], 'dialogue_safety', 'wiki-toxic-comments'
        )

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.build(opt)
            self._setup_data(opt['datatype'])

        self.label_candidates = [NOT_OK_CLASS, OK_CLASS]

        opt = copy.deepcopy(opt)
        super().__init__(opt, shared)
        self.reset()

    def _get_data(self):
        # useful constants
        # all of these colors are bolded
        RESET = '\033[0m'
        RED = '\033[1;91m'
        YELLOW = '\033[1;93m'
        GREEN = '\033[1;92m'
        BLUE = '\033[1;96m'
        CYAN = '\033[1;94m'
        MAGENTA = '\033[1;95m'

        # only use colors if we're outputting to a terminal
        USE_COLORS = _sys.stdout.isatty()
        if not USE_COLORS:
            RESET = RED = YELLOW = GREEN = BLUE = CYAN = MAGENTA = ''

        # generate the rainbow stars
        rainbow = [RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA]
        size = 78 // len(rainbow)
        stars = ''.join([color + '*' * size for color in rainbow])
        stars += RESET

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.isfile(os.path.join(self.data_path, 'train.csv')):
            raise RuntimeError(
                f'\n\n{stars}\nThis data must be downloaded from '
                '<https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data>. '
                '\nIt cannot be automatically downloaded, as one must agree to '
                'the competition rules outlined on the website before '
                'gaining access to the data.\n\n'
                'Once downloaded, please put the data in the following '
                f'directory: \n{self.data_path}\n{stars}'
            )

    def data_to_json(self, pd, file_name):
        response = pd.to_dict('records')
        with open(os.path.join(self.data_path, file_name), 'w') as f:
            f.write(json.dumps(response, indent=4))

    def build(self, opt):
        self._get_data()

        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Please install pandas by running `pip install pandas`')

        version = 'v1.0'
        read_path = self.data_path
        if not build_data.built(self.data_path, version):
            print('[building data from : ' + read_path + ']')
            build_data.make_dir(self.data_path)
            # Read in data
            train = pd.read_csv(os.path.join(read_path, 'train.csv'))
            test = pd.read_csv(os.path.join(read_path, 'test.csv'))
            test_labels = pd.read_csv(os.path.join(read_path, 'test_labels.csv'))

            # Labels for test data; value of -1 indicates it was not used for scoring
            test_labels = test_labels[
                (test_labels.toxic != -1)
                & (test_labels.severe_toxic != -1)
                & (test_labels.obscene != -1)
                & (test_labels.threat != -1)
                & (test_labels.insult != -1)
                & (test_labels.identity_hate != -1)
            ]

            # Merge test data with labels
            test = pd.merge(test_labels, test, on='id')

            # Split 10% of train data to be valid
            test['data_type'] = 'test'
            train['data_type'] = 'train'
            valid_set = train.sample(frac=0.1, random_state=42)
            valid_set['data_type'] = 'valid'
            train.update(valid_set)

            # Combine test and train into one data frame
            total_data = pd.concat([test, train], ignore_index=True, sort=False)
            # Rename comment_text to text for the act dict
            total_data.rename(columns={'comment_text': 'text'}, inplace=True)

            # Use the different categories to get binary classification
            total_data['sensitive'] = (
                total_data['severe_toxic']
                + total_data['toxic']
                + total_data['obscene']
                + total_data['threat']
                + total_data['insult']
                + total_data['identity_hate']
            )

            total_data.loc[total_data['sensitive'] < 1, 'is_sensitive'] = 0
            total_data.loc[total_data['sensitive'] >= 1, 'is_sensitive'] = 1

            # Drop unecessary column
            total_data = total_data.drop(columns=['id'])

            self.data_to_json(total_data, 'wiki-toxic-comments-default.json')

            # Partition 80/10/10 according to arXiv:1811.12900 [cs.CL]
            # 90 train 10 test in paper
            # split based on command line flag
            original_train = total_data[
                (total_data['data_type'] == 'valid')
                | (total_data['data_type'] == 'train')
            ].copy()
            l_td = len(original_train)

            original_train.iloc[: int(0.8 * l_td)]['data_type'] = 'train'
            original_train.iloc[int(0.8 * l_td) : int(0.9 * l_td)]['data_type'] = 'test'
            original_train.iloc[int(0.9 * l_td) :]['data_type'] = 'valid'

            self.data_to_json(original_train, 'wiki-toxic-comments-partition.json')

            # Merge the 3 files to get a list of dicts as follows:
            # [
            #     {
            #         'toxic': 0 or 1 ,
            #         'severe_toxic': 0 or 1,
            #         'obscene': 0 or 1,
            #         'threat': 0 or 1,
            #         'insult': 0 or 1,
            #         'identity_hate': 0 or 1,
            #         'text': <comments>,
            #         'data_type': test/validation/train,
            #         'sensitive': 0.0,
            #         'is_sensitive': 0/1
            #     },
            #   ...
            # ]

            build_data.mark_done(self.data_path, version)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def _balance_data(self, data_list):
        # NOTE: this assumes that there are more OK labels than NOT OK labels
        ok = [x for x in data_list if x['is_sensitive'] == 0]
        notok = [x for x in data_list if x['is_sensitive'] == 1]

        new_not_ok = []
        while len(new_not_ok) < len(ok):
            new_not_ok.append(self.fixed_random.choice(notok))

        new_data = ok + new_not_ok
        self.fixed_random.shuffle(new_data)
        return new_data

    def _setup_data(self, datatype):
        """
        Set up the data based on the correct partition flag specified and partition
        accordingly.
        """
        if not self.use_test_set:
            dp = os.path.join(self.data_path, 'wiki-toxic-comments-partition.json')
        else:
            dp = os.path.join(self.data_path, 'wiki-toxic-comments-default.json')

        print('loading: ' + dp)
        with open(dp, 'r') as f:
            self.total_data = json.loads(f.read())
            if 'train' in datatype:
                self.data = [x for x in self.total_data if x['data_type'] == 'train']
            elif 'test' in datatype:
                self.data = [x for x in self.total_data if x['data_type'] == 'test']
            else:
                self.data = [x for x in self.total_data if x['data_type'] == 'valid']

        if self.balance_data and 'train' in datatype:
            self.data = self._balance_data(self.data)

    def get(self, episode_idx, entry_idx):
        sample = self.data[episode_idx]
        sample['label_candidates'] = self.label_candidates
        sample['episode_done'] = True
        sample['labels'] = [self.label_candidates[int(sample['is_sensitive']) - 1]]
        return Message(sample)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DefaultTeacher(WikiToxicCommentsTeacher):
    pass
