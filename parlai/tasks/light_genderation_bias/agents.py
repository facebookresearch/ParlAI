#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher

from parlai.tasks.light_dialog.agents import DefaultTeacher as OrigLightTeacher
from parlai.tasks.light_genderation_bias.build import build
from parlai.utils.io import PathManager

from collections import deque
from copy import deepcopy
import csv
import json
import os
import random


# Filenames
NEW_DATA = 'new_data_dialogue_only.json'
GENDERED_LIST = 'gendered_list.tsv'
COUNTERFACTUALS = 'counterfactuals.json'


def _path(opt):
    build(opt)
    datapath = opt['datapath']
    return os.path.join(datapath, 'light_genderation_bias')


PUNCTUATION_LST = [
    (' .', '.'),
    (' !', '!'),
    (' ?', '?'),
    (' ,', ','),
    (" ' ", "'"),
    (" . . . ", "... "),
    (" ( ", " ("),
    (" ) ", ") "),
    (" ; ", "; "),
]


##############################################################
## UTILITY FUNCTIONS
##############################################################


def get_finegrained_count(text, gender_dct):
    """
    Count the number of female, male, and neutral gendered words in a string, given the
    gender dict.
    """
    text = text.lower()
    f_count = 0
    m_count = 0
    n_count = 0
    for line in text.split('\n'):
        words = line.split(' ')
        for word in words:
            if word in gender_dct:
                if gender_dct[word]['gender'] == 'F':
                    f_count += 1
                else:
                    m_count += 1
            else:
                n_count += 1

    return f_count, m_count, n_count


def format_text(text):
    text = text.lower()
    for punc in PUNCTUATION_LST:
        text = text.replace(punc[1], punc[0])

    return text


def unformat_text(text):
    for punc in PUNCTUATION_LST:
        text = text.replace(punc[0], punc[1])

    return text


def read_gender_tsv(path, remove_verbs=True):
    """
    Load TSV of gendered word lists and return a dict.
    """
    gender_dct = {}
    with PathManager.open(path) as tsvfile:
        reader = list(csv.reader(tsvfile, delimiter='\t'))
        title_lst = reader[0]
        title_dict = {}
        for idx, title in enumerate(title_lst):
            title_dict[idx] = title

        for i in range(1, len(reader)):
            row = reader[i]
            word = row[0].lower()
            gender_dct[word] = {}
            for j, category in enumerate(row[1:]):
                gender_dct[word][title_dict[j + 1]] = category

    if remove_verbs:
        return {k: v for k, v in gender_dct.items() if v['syncategory'] != 'verb'}

    return gender_dct


def flatten(episode, context_length, include_labels=True, delimiter='\n'):
    """
    Flatten the data into single example episodes.

    This is used to make conditional training easier and for a fair comparison of
    methods.
    """
    context = deque(maxlen=context_length if context_length > 0 else None)
    new_episode = []

    for ex in episode:
        context.append(ex.get('text', ''))
        # add context
        if len(context) > 1:
            ex.force_set('text', delimiter.join(context))
        # set episode_done to be True
        ex.force_set('episode_done', True)
        labels = ex.get('labels', ex.get('eval_labels', None))
        if labels is not None and include_labels:
            context.append(random.choice(labels))

        new_episode.append(ex)

    return new_episode


##############################################################
## TEACHERS FUNCTIONS
##############################################################
class LightGenderTeacher(FixedDialogTeacher):
    """
    ALL LIGHT Teacher: combines all gender bias mitigation methods described in.

    <https://arxiv.org/abs/1911.03842>.

    This teacher combines methods for gender bias mitigation in
    dialogue generation on the LIGHT dialogue task, including:
    - Adding more positive data (Pos Data):`--add-new-data`
    - Conditional training (Bias Ctrl): `--add-conditional`
    - Counterfactual Data Augmentation (CDA): `--add-counterfactual`

    For more information, please see our projects page:
    <https://parl.ai/projects/genderation_bias/>
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('LIGHT Gender Args')
        OrigLightTeacher.add_cmdline_args(parser)
        parser.add_argument(
            '--add-new-data',
            type='bool',
            default=True,
            help='Add the extra data collected',
        )
        parser.add_argument(
            '--add-conditional',
            type='bool',
            default=True,
            help='Append the conditional bucket to the end of the input or not',
        )
        parser.add_argument(
            '--add-counterfactual',
            type='bool',
            default=True,
            help='Load counterfactual data for training',
        )
        parser.add_argument(
            '--force-conditional',
            type=str,
            default='none',
            help='Add one bucket to everything, regardless of the true bin',
        )
        parser.add_argument(
            '--bucket-only',
            type=str,
            default='none',
            help='Only train/evaluate on examples from a specific bucket',
        )

        parser.set_defaults(
            light_use_objects=False,
            light_use_setting=False,
            light_use_affordances=False,
            light_use_persona_names=False,
            light_use_persona='none',
            light_use_action='none',
            light_use_emote='none',
            use_reply='none',
        )
        return parser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.gender_dict = read_gender_tsv(os.path.join(_path(opt), GENDERED_LIST))
        self.fixed_random = random.Random(42)

        self.add_conditional = opt['add_conditional']
        self.add_counterfactual = opt['add_counterfactual']
        self.add_new_data = opt['add_new_data']

        self.force_conditional = opt['force_conditional']
        self.bucket_only = opt['bucket_only']
        if self.force_conditional == 'none':
            self.force_conditional = None
        if self.bucket_only == 'none':
            self.bucket_only = None

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.data = self._setup_data(opt)

        super().__init__(opt, shared)
        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def get_bucket(self, text):
        """
        Get the bin or "bucket" that a string falls into.

        The bucket is determined based on the presence of male or femael gendered words
        in the text.
        """
        text = format_text(text)
        # we're bucketing based on the number of gendered words
        f_count, m_count, n_count = get_finegrained_count(text, self.gender_dict)
        # get f bucket
        if f_count == 0:
            f_bucket = 'f0'
        else:
            f_bucket = 'f1'

        # get m bucket
        if m_count == 0:
            m_bucket = 'm0'
        else:
            m_bucket = 'm1'

        # combine bucket
        all_bucket = ' '.join([f_bucket, m_bucket])
        return all_bucket

    def _flip_str(self, txt_lines):
        new_lines = []
        lines = txt_lines.split('\n')
        for text in lines:
            f_text = format_text(text)
            f_text_lst = f_text.split(' ')
            new_words = []
            for word in f_text_lst:
                if word in self.swap_dct:
                    if word == 'her':
                        # silly heuristic: choose him/his 50% of the time
                        random_choice = random.choice([0, 1])
                        if random_choice:
                            new_word = 'his'
                        else:
                            new_word = 'him'
                    else:
                        new_word = self.swap_dct[word]['word']
                else:
                    new_word = word
                new_words.append(new_word)
            new_f_text = ' '.join(new_words)
            uf_text = unformat_text(new_f_text)
            new_lines.append(uf_text)

        return '\n'.join(new_lines)

    def _flip_ex(self, ex):
        """
        Return the counterfactual example for a given example (i.e. swap 'he' --> 'she')
        """
        new_ex = deepcopy(ex)
        text = ex['text']
        labels = 'labels' if 'labels' in ex else 'eval_labels'
        label = ex[labels][0]
        new_ex.force_set('text', self._flip_str(text))
        new_ex.force_set(labels, [self._flip_str(label)])
        return new_ex

    def _get_new_data(self, opt):
        """
        Load extra positive dialogue data IFF datatype==train.
        """
        dt = opt['datatype'].split(':')[0]
        if dt == 'train':
            with PathManager.open(os.path.join(_path(opt), NEW_DATA), 'r') as f:
                data = json.load(f)
            new_data = []
            for ep in data:
                new_ep = []
                for ex in ep:
                    ex['new_data'] = True
                    new_ep.append(Message(ex))
                new_data.append(new_ep)
            return new_data

        return []

    def _setup_data(self, opt):
        """
        Load original LIGHT dataset.
        """
        # Add new data?
        dt = opt['datatype'].split(':')[0]

        orig_episodes = OrigLightTeacher(opt).episodes
        if self.add_new_data:
            new_data = self._get_new_data(opt)
            total_data = orig_episodes + new_data
            self.fixed_random.shuffle(total_data)
            orig_episodes = total_data

        # Flatten this data
        flat_episodes = []
        for ep in orig_episodes:
            # flatten the episode into 1-example episodes with context
            flattened_ep = flatten(ep, -1, include_labels=True, delimiter='\n')
            flat_episodes += flattened_ep

        # Counterfactual?
        if self.add_counterfactual and dt != 'test':
            with PathManager.open(os.path.join(_path(opt), COUNTERFACTUALS), 'rb') as f:
                self.swap_dct = json.load(f)

            new_episodes = []
            for ex in flat_episodes:
                new_ex = self._flip_ex(ex)
                ex['counterfactual'] = False  # mark which episode is swapped
                new_ex['counterfactual'] = True
                # add both old and new examples
                new_episodes.append(ex)
                new_episodes.append(new_ex)

            flat_episodes = new_episodes

        # Conditional training?
        bucket_percentages = {}
        new_episodes = []
        for ex in flat_episodes:
            label_type = 'labels' if 'labels' in ex else 'eval_labels'
            label = ex[label_type][0]
            # get bucket for label
            bucket_key = self.get_bucket(label)
            # update the bucket percentages
            bucket_percentages.setdefault(bucket_key, 0)
            bucket_percentages[bucket_key] += 1
            # append this bucket to the text field
            if self.add_conditional:
                if self.force_conditional is None:
                    new_text = ex['text'] + '\n' + bucket_key
                else:
                    # force the model to see a specific bucket every time
                    # NOTE: we still track the original bucket that the
                    # text was supposed to fall into
                    new_text = ex['text'] + self.force_conditional
                ex.force_set('text', new_text)
            ex['bucket'] = bucket_key
            if self.bucket_only is None or self.bucket_only == bucket_key:
                new_episodes.append(ex)

        # Summarize the bucket distribution
        print('Distribution of bins:')
        total = sum(bucket_percentages.values())
        strs = []
        for k, v in bucket_percentages.items():
            pct = round((v / total) * 100, 2)
            strs.append(f'{k}: {pct}%')
        strs = sorted(strs)
        for string in strs:
            print(string)

        return new_episodes

    def get(self, episode_idx, entry_idx=0):
        return Message(self.data[episode_idx])

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DefaultTeacher(LightGenderTeacher):
    pass
