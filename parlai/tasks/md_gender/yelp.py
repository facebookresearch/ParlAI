#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
import parlai.tasks.md_gender.utils as gend_utils
import parlai.utils.logging as logging

from copy import deepcopy
import os
import random
import sys as _sys


class YelpTeacher(FixedDialogTeacher):
    """
    Yelp MD Gender Teacher.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train
        self.label_candidates = gend_utils.ALL_CANDS

        if shared is None:
            # set map
            self.data = self._setup_data(opt)
        else:
            self.data = shared['data']
        super().__init__(opt, shared)
        self.reset()

    def _check_data_downloaded(self, opt):
        # Checks whether the data is downloaded properly
        # Also checks whether data is built, and builds it if so
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

        self.data_path = os.path.join(opt['datapath'], 'md_gender', 'yelp')
        if not os.path.exists(self.data_path):
            PathManager.mkdirs(self.data_path)
        if not PathManager.exists(
            os.path.join(self.data_path, 'valid.fader.with_cat.40000')
        ):
            raise RuntimeError(
                f'\n\n{stars}\nThis data must be downloaded following instructions in '
                'the README here:'
                '<https://github.com/facebookresearch/MultipleAttributeTextRewriting/blob/master/data/README.md>. '
                '\nIt cannot be automatically downloaded, as one must agree to '
                'the terms outlined on the website before gaining access to the data.\n\n'
                'Once downloaded, please put the data in the following '
                f'directory: \n{self.data_path}\n{stars}'
            )
        elif not PathManager.exists(os.path.join(self.data_path, 'classtrain.txt')):
            logging.info('[ Building data ... ]')
            # build train
            with open(os.path.join(self.data_path, 'classtrain.txt'), 'w') as f:
                for fle_num in [4000, 6000, 8000]:
                    train_fle = f'train.fader.with_cat.{fle_num}'
                    with open(os.path.join(self.data_path, train_fle)) as g:
                        lines = g.readlines()
                        for line in lines:
                            tabs = line.split('\t')
                            text = tabs[0]
                            gend = tabs[1]
                            if gend == '0':
                                f.write(f'male\t{text}\n')
                            elif gend == '1':
                                f.write(f'female\t{text}\n')

            # build valid and test
            for pair in [('dev', 'valid'), ('test', 'test')]:
                with open(
                    os.path.join(self.data_path, f'female_only.{pair[0]}.en'), 'w'
                ) as fem_val:
                    with open(
                        os.path.join(self.data_path, f'male_only.{pair[0]}.en'), 'w'
                    ) as masc_val:
                        for fle_num in [4000, 6000, 8000]:
                            valid_fle = f'{pair[1]}.fader.with_cat.{fle_num}'
                            with open(
                                os.path.join(self.data_path, valid_fle), 'r'
                            ) as g:
                                lines = g.readlines()
                                for line in lines:
                                    tabs = line.split('\t')
                                    text = tabs[0]
                                    gend = tabs[1]
                                    if gend == '0':
                                        masc_val.write(f'{text}\n')
                                    elif gend == '1':
                                        fem_val.write(f'{text}\n')

    def _load_gender_data(self, datatype):
        """
        Load data from the checkpoint.
        """
        dt = datatype.split(':')[0]
        data = []
        folder = self.data_path
        if dt == 'train':
            gender_cnt = {gend_utils.MASC: 0, gend_utils.FEM: 0}
            fle = os.path.join(folder, 'classtrain.txt')
            with open(fle, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    gender, text = line.split('\t')
                    data.append(
                        {
                            'text': text,
                            'labels': [f'SELF:{gender}'],
                            'class_type': 'self',
                        }
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
                        {
                            'text': f_line,
                            'labels': [f'SELF:{gend_utils.FEM}'],
                            'class_type': 'self',
                        }
                    )
                    data.append(
                        {
                            'text': m_line,
                            'labels': [f'SELF:{gend_utils.MASC}'],
                            'class_type': 'self',
                        }
                    )

        return data

    def _setup_data(self, opt):
        # check that the data was downloaded and set up properly
        self._check_data_downloaded(opt)
        # Load map from image ID to gender
        data = self._load_gender_data(opt['datatype'])

        extra_data = []
        if self.add_unknown_classes:
            # load about data (unknown but inferred)
            extra_data = gend_utils.get_inferred_about_data(
                self.opt['task'], self.opt['datatype']
            )

            # now create partner/TO data: true neutral
            for ex in data:
                partner_ex = deepcopy(ex)
                partner_ex['labels'] = [f'PARTNER:{gend_utils.NEUTRAL}']
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
