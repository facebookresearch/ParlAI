#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Dialogue safety related datasets and teachers.
"""

# from tasks.civil_bias_toxic_comment.wikiagents import DefaultTeacher
from typing import Optional

from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import parlai.core.build_data as build_data
from parlai.tasks.dialogue_safety.agents import WikiToxicCommentsTeacher
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.utils.io import PathManager

import copy
import os
import random
import json

GENDER = ['male', 'female', 'transgender', 'other_gender']
SEXUAL_ORIENTATION = [
    'heterosexual',
    'homosexual_gay_or_lesbian',
    'bisexual',
    'other_sexual_orientation',
]
RELIGION = [
    'christian',
    'jewish',
    'muslim',
    'hindu',
    'buddhist',
    'atheist',
    'other_religion',
]
ETHNICITY = ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
DISABILITY = [
    'physical_disability',
    'intellectual_or_learning_disability',
    'psychiatric_or_mental_illness',
    'other_disability',
]

BIAS_TYPES = {
    'gender': GENDER,
    'sexual_orientation': SEXUAL_ORIENTATION,
    'religion': RELIGION,
    'ethnicity': ETHNICITY,
    'disability': DISABILITY,
}

ALL_BIAS_LABELS = GENDER + SEXUAL_ORIENTATION + RELIGION + ETHNICITY + DISABILITY


class CivilBiasToxicityTeacher(WikiToxicCommentsTeacher):
    """
    Dataset of comments from the Civil Comments platform. Taken from the Jigsaw
    Unintended Bias in Toxicity Classification on Kaggle.

    <https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data>

    We convert this data to a binary classification task.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group(
            'Civil Bias and Toxic Comments Classification Data'
        )
        parser.add_argument(
            '--data-threshold',
            type=float,
            default=0.5,
            help='The threshold to determine if a comment is safe or not;'
            'ranges from (0.0, 1.0), and by default is 0.5.',
        )
        parser.add_argument(
            '--bias-subset',
            type='bool',
            default=False,
            help='Defaults to using the whole dataset. If value is set to '
            'true, then we will just look at the subset that has the gender,'
            'religion, and disability labels.',
        )
        return parser

    def __init__(self, opt, shared=None):
        self.fixed_random = random.Random(42)
        self.use_test_set = opt['use_test_set']
        self.balance_data = opt['balance_data']

        if opt['data_threshold'] > 0 and opt['data_threshold'] < 1:
            self.data_threshold = opt['data_threshold']
        else:
            self.data_threshold = 0.5

        # used to decide whether to consider the smaller subset
        self.bias_subset = opt['bias_subset']
        self.DATA_SOURCE = '<https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data>'
        # different columns of the data; could be useful later
        self.meta_data = [
            'id',
            'created_date',
            'publication_id',
            'parent_id',
            'article_id',
            'rating',
            'funny',
            'wow',
            'sad',
            'likes',
            'disagree',
            'identity_annotator_count',
            'toxicity_annotator_count',
        ]

        # these ratings have all labels
        self.rating = [
            'toxicity',
            'severe_toxicity',
            'obscene',
            'identity_attack',
            'sexual_explicit',
            'insult',
            'threat',
        ]
        self.default_str = 'civil-bias-toxic-comments-default-'
        self.original_train_str = 'civil-bias-toxic-comments-original-train-'
        self.data_path = os.path.join(opt['datapath'], 'civil-bias-toxic-comments')

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.build(opt)
            self._setup_data(opt['datatype'])

        self.label_candidates = [NOT_OK_CLASS, OK_CLASS]

        opt = copy.deepcopy(opt)
        super(WikiToxicCommentsTeacher, self).__init__(opt, shared)
        self.reset()

    def build(self, opt):
        # check if the datasets exist
        self._get_data()

        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Please install pandas by running `pip install pandas`')

        version = 'v0.5'
        read_path = self.data_path
        if not build_data.built(read_path, version):
            print('[building data from : ' + read_path + ']')
            build_data.make_dir(read_path)
            # Read in data
            train = pd.read_csv(os.path.join(read_path, 'train.csv'))
            # train.rename(columns={'target': 'toxicity'}, inplace=True)
            # test.csv and test_private_expanded.csv should have the same comments,
            # so instead of reading both, just reading test.csv
            test = pd.read_csv(os.path.join(read_path, 'test_private_expanded.csv'))
            train.rename(columns={'target': 'toxicity'}, inplace=True)

            # set the data types
            # Split 10% of train data to be valid
            test['data_type'] = 'test'
            train['data_type'] = 'train'
            valid_set = train.sample(frac=0.1, random_state=self.fixed_random)
            valid_set['data_type'] = 'valid'
            train.update(valid_set)

            # Combine test and train into one data frame
            total_data = pd.concat([test, train], ignore_index=True, sort=False)
            # Rename comment_text to text for the actual json file
            total_data.rename(columns={'comment_text': 'text'}, inplace=True)

            # Drop unecessary column
            total_data = total_data.drop(columns=self.meta_data)

            # create max_rating column used later for things
            total_data['max_rating'] = total_data[self.rating_columns].max(axis=1)
            total_data['is_sensitive'] = total_data['max_rating'] > 0.5

            # create column for subset of labelled_bias
            total_data['bias_labelled'] = True
            row_selector = total_data[self.bias_labels].isnull().T.any()
            total_data.loc[row_selector, 'bias_labelled'] = False

            # default split
            self.data_to_json_by_partition(total_data, self.default_str)

            # Partition 80/10/10 according to arXiv:1811.12900 [cs.CL]
            # 90 train 10 test in paper
            # split based on command line flag
            original_train = total_data[
                (total_data['data_type'] == 'valid')
                | (total_data['data_type'] == 'train')
            ].copy()
            l_td = len(original_train)

            # make sure the index is good
            original_train.reset_index(drop=True)
            original_train.loc[: int(0.8 * l_td), 'data_type'] = 'train'
            original_train.loc[int(0.8 * l_td) : int(0.9 * l_td), 'data_type'] = 'test'
            original_train.loc[int(0.9 * l_td) :, 'data_type'] = 'valid'

            self.data_to_json_by_partition(original_train, self.original_train_str)

            build_data.mark_done(self.data_path, version)
            print("finished building all data")

    def data_to_json_by_partition(self, total_data, start_string):
        partitions = ['train', 'test', 'valid']
        for partition in partitions:
            parition_data = total_data[total_data['data_type'] == partition]
            print(start_string + partition + '.json', len(parition_data))
            self.data_to_json(parition_data, start_string + partition + '.json')

    def _setup_data(self, datatype):
        """
        Set up the data based on the correct partition flag specified and partition
        accordingly.
        """
        if self.use_test_set:
            dp = os.path.join(self.data_path, self.default_str)
        else:
            dp = os.path.join(self.data_path, self.original_train_str)

        # loading by data type
        if 'train' in datatype:
            self._load_data(dp + 'train.json')
        elif 'test' in datatype:
            self._load_data(dp + 'test.json')
        else:
            self._load_data(dp + 'valid.json')

        # checking if bias subset
        if self.bias_subset:
            self.data = [x for x in self.data if x['bias_labelled']]

        # relabelling based on threshold if necessary
        if self.data_threshold != 0.5:
            for x in self.data:
                x['is_sensitive'] = x['max_rating'] > self.data_threshold

        # balance data if necessary
        if self.balance_data and 'train' in datatype:
            print("balancing data")
            self.data = self._balance_data(self.data)

    def _load_data(self, fname):
        print('loading: ' + fname)
        with PathManager.open(fname, 'r') as f:
            self.data = json.loads(f.read())


class DefaultTeacher(CivilBiasToxicityTeacher):
    pass
