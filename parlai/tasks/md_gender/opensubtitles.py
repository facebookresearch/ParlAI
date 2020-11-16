#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher, FbDeprecatedDialogTeacher
import parlai.tasks.md_gender.utils as gend_utils
from parlai.tasks.md_gender.build_opensubtitles import build

from copy import deepcopy
import os
import random
import re


def get_gender_data():
    """
    Load a dict that maps the name to the probability of the gender.
    """
    names = gend_utils.get_name_gender_cnts()

    # Now convert counts to probabilities and lowercase names
    new_names = {}
    for name, dct in names.items():
        tot = sum(dct.values())
        new_names[name.lower()] = {
            'M': round(dct['M'] / tot, 3),
            'F': round(dct['F'] / tot, 3),
        }
    names = new_names

    # Load all explicitly gendered words
    male_words, female_words = gend_utils.get_explicitly_gendered_words()

    for word in male_words:
        names[word.lower()] = {'M': 1.0, 'F': 0.0}

    for word in female_words:
        names[word.lower()] = {'M': 0.0, 'F': 1.0}

    return names


def get_gender(name_str, dct, threshold=0.7):
    name = name_str.lower()
    name = re.sub("[^A-Za-z]", "", name)
    if name not in dct:
        return gend_utils.UNKNOWN

    gender_dct = dct[name]
    max_prob = max(gender_dct.values())
    if max_prob > threshold:
        if max_prob == gender_dct['F']:
            return gend_utils.FEM
        else:
            return gend_utils.MASC
    else:
        return gend_utils.UNKNOWN


class OpensubtitlesTeacher(FixedDialogTeacher):
    """
    Opensubtitles gender teacher.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        agent = argparser.add_argument_group('Opensubtitles gender')
        agent.add_argument(
            '--labels-to-use',
            type=str,
            default='all',
            choices=['all', 'self', 'partner'],
            help='Which labels to use for this teacher',
        )

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train
        self.labels_to_use = opt['labels_to_use']
        self.label_candidates = gend_utils.ALL_CANDS
        if shared is None:
            # set map
            self.data = self._setup_data(opt)
            if (self.is_train and opt['balance']) or (
                self.is_valid and opt['balance_valid']
            ):
                to_exclude = gend_utils.ABOUT_CANDS
                self.data = gend_utils.balance_data(
                    self.data, exclude_labels=to_exclude
                )
        else:
            self.data = shared['data']
        super().__init__(opt, shared)
        self.reset()

    def _load_orig_data(self, opt):
        """
        Load the original data, combining all consecutive examples from a single speaker
        into one example.

        Return a list of episodes.
        """
        dt = opt['datatype'].split(':')[0]
        if dt == 'train':
            dt = 'train:evalmode'
        teach_opt = deepcopy(opt)
        teach_opt['datatype'] = dt
        orig_teach = OpensubtitlesFilteredDialogTeacher(teach_opt)
        # now organize by episode and place dialogues with
        # the same ID together
        episodes = []
        curr_episode = []
        curr_ex = {'id': None, 'text': ''}
        ep_done = False
        prev_speaker = None

        while True:
            ex, epoch_done = orig_teach.next_example()
            ex_id = ex['text'].split('): ', 1)[0][1:]
            ex_text = ex['text'].split('): ', 1)[1]
            ep_done = ex['episode_done']

            if ex_id == prev_speaker:
                curr_ex['text'] += ' ' + ex_text
            else:
                if curr_ex['id'] is not None:
                    curr_episode.append(curr_ex)
                curr_ex = {'id': ex_id, 'text': ex_text}
                prev_speaker = ex_id

            if 'labels' in ex:
                if len(ex['labels']) == 1:
                    label_id = ex['labels'][0].split('): ', 1)[0][1:]
                    label_text = ex['labels'][0].split('): ', 1)[1]

                    if label_id == prev_speaker:
                        curr_ex['text'] += ' ' + label_text
                    else:
                        if curr_ex['id'] is not None:
                            curr_episode.append(curr_ex)
                        curr_ex = {'id': label_id, 'text': label_text}
                        prev_speaker = label_id

            if ep_done:
                if curr_ex['id'] is not None:
                    curr_episode.append(curr_ex)
                episodes.append(curr_episode)
                curr_ex = {'id': None, 'text': ''}
                curr_episode = []
                ep_done = False
                prev_speaker = None

            if epoch_done:
                break

        return episodes

    def _setup_data(self, opt):
        """
        Load the raw data and annotate it with gender.
        """
        raw_data = self._load_orig_data(opt)
        gender_data = get_gender_data()
        data = []

        counts = {
            'self': {gend_utils.MASC: 0, gend_utils.FEM: 0, gend_utils.UNKNOWN: 0},
            'partner': {gend_utils.MASC: 0, gend_utils.FEM: 0, gend_utils.UNKNOWN: 0},
        }

        for episode in raw_data:
            for i, ex in enumerate(episode):
                self_gender = get_gender(ex['id'], gender_data)
                if len(episode) > i + 1:
                    # peak into the future to see the gender of
                    # the person they are speaking
                    partner_id = episode[i + 1]['id']
                    partner_gender = get_gender(partner_id, gender_data)
                elif i > 0:
                    # look back to see who they are replying to
                    partner_id = episode[i - 1]['id']
                    partner_gender = get_gender(partner_id, gender_data)

                else:
                    # episode is length 1, no partner
                    partner_id = None
                    partner_gender = gend_utils.UNKNOWN

                # Collect stats
                counts['self'][self_gender] += 1
                counts['partner'][partner_gender] += 1

                if self_gender is not None and self.labels_to_use != 'partner':
                    labels = [f'SELF:{self_gender}']
                    data.append(
                        {
                            'text': ex['text'],
                            'name': ex['id'],
                            'partner_name': partner_id,
                            'self_gender': self_gender,
                            'partner_gender': partner_gender,
                            'labels': labels,
                            'class_type': 'self',
                        }
                    )
                if partner_gender is not None and self.labels_to_use != 'self':
                    labels = [f'PARTNER:{partner_gender}']
                    data.append(
                        {
                            'text': ex['text'],
                            'name': ex['id'],
                            'partner_name': partner_id,
                            'self_gender': self_gender,
                            'partner_gender': partner_gender,
                            'labels': labels,
                            'class_type': 'partner',
                        }
                    )

        if self.labels_to_use == 'all' and self.add_unknown_classes:
            # load about data
            all_about_data = gend_utils.get_inferred_about_data(
                self.opt['task'], self.opt['datatype']
            )
            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                # do something here
                to_samp = int(sample_rate * len(all_about_data))
                sampled = random.sample(all_about_data, to_samp)
                data += sampled
            else:
                data += all_about_data

        if self.is_train:
            random.shuffle(data)

        # Print data distribution
        for x in ['self', 'partner']:
            print(f'\nFor {x.upper()}:')
            tot = sum(counts[x].values())
            print(f'Total: {tot}')
            for k, v in counts[x].items():
                print(f'{k}: {v} ({round(v / tot, 3)})')

        return data

    def get(self, episode_idx, entry_idx=0):
        ex = self.data[episode_idx]
        text = ex['text']
        labels = ex['labels']
        class_type = ex['class_type']
        if gend_utils.UNKNOWN in labels[0]:
            # During training we flip between labels
            labels = gend_utils.UNKNOWN_LABELS[class_type]
        return Message(
            {
                'text': text,
                'labels': labels,
                'episode_done': True,
                'label_candidates': self.label_candidates[class_type],
                'id': 'Opensubtitles Gender',
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


class OpensubtitlesFilteredDialogTeacher(FbDeprecatedDialogTeacher):
    """
    Raw data teacher, pre-filtered by @Ledell.
    """

    def __init__(self, opt, shared=None):
        dt = opt['datatype'].split(':')[0]
        datapath = opt['datapath']
        if shared is None:
            build(datapath)
        fle = os.path.join(datapath, 'md_gender', 'opensubtitles', f'{dt}.txt.filtered')
        opt['datafile'] = fle
        super().__init__(opt, shared)
