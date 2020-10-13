#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import ChunkTeacher
from parlai.tasks.wikipedia.build import build
from parlai.utils.misc import warn_once
import parlai.tasks.md_gender.utils as gend_utils

from copy import deepcopy
import json
import os
import random
import re
import spacy
from typing import List


NLP = spacy.load('en_core_web_sm')

DEBUG = False


def check_if_person(text):
    """
    Using spacy, check if the title of the Wikipedia passage is a person.
    """
    doc = NLP(text)
    is_person = False
    ents = [ent for ent in doc.ents]
    for ent in ents:
        if ent.label_ == 'PERSON':
            is_person = True
    return is_person


def get_gender(text):
    """
    Determine gender by the count of referring pronouns in the biography (he, she,
    they).
    """
    he_count = len(re.findall(' he ', text.lower()))
    she_count = len(re.findall(' she ', text.lower()))
    they_count = len(re.findall(' they ', text.lower()))

    if he_count == max(he_count, she_count, they_count):
        return gend_utils.MASC
    elif she_count == max(he_count, she_count, they_count):
        return gend_utils.FEM
    else:
        nonbinary_count = len(re.findall(' non-binary ', text.lower()))
        if nonbinary_count > 0:
            return gend_utils.NONBINARY
        return gend_utils.NEUTRAL


class WikipediaTeacher(ChunkTeacher):
    """
    Wikipedia gender.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        agent = argparser.add_argument_group('Wiki gender')
        agent.add_argument(
            '--class-task',
            type=str,
            default='all',
            choices=['single', 'all'],
            help='Rank against all possibilities vs. F/M/N/NB',
        )
        agent.add_argument(
            '--mask-gendered-words',
            type='bool',
            default=False,
            help='Mask explicitly gendered words.',
        )
        return argparser

    def __init__(self, opt, shared=None):
        if shared is None:
            # set map
            self.opt = opt
            self._set_chunk_idx_to_file(opt)
        else:
            self.chunk_idx_to_file = shared['chunk_idx_to_file']

        self.class_task = opt['class_task']
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.add_unknown_classes = (
            opt['add_unknown_classes'] and self.is_train and self.class_task == 'all'
        )

        if self.class_task == 'all':
            self.label_candidates = gend_utils.ALL_CANDS
        elif self.class_task == 'single':
            self.label_candidates = {
                'about': [
                    gend_utils.FEM,
                    gend_utils.MASC,
                    gend_utils.NEUTRAL,
                    gend_utils.NONBINARY,
                ]
            }

        self.mask_gendered_words = opt['mask_gendered_words']
        if self.mask_gendered_words:
            male, female = gend_utils.get_explicitly_gendered_words(self.opt)
            self.gendered_list = male + female

        self.counts = {
            gend_utils.FEM: 0,
            gend_utils.MASC: 0,
            gend_utils.NEUTRAL: 0,
            gend_utils.NONBINARY: 0,
        }
        super().__init__(opt, shared)

    def _set_chunk_idx_to_file(self, opt):
        # download wikipedia data
        wiki_opt = deepcopy(opt)
        wiki_opt['task'] = 'wikipedia:full'
        build(wiki_opt)

        # now divide data into subfolders
        data_folder = self._get_data_folder()
        self.chunk_idx_to_file = {}
        i = 0
        for subdir in os.listdir(data_folder):
            if subdir != 'README.md':
                for fle in os.listdir(os.path.join(data_folder, subdir)):
                    self.chunk_idx_to_file[i] = os.path.join(data_folder, subdir, fle)
                    i += 1

    def _get_data_folder(self):
        return os.path.join(self.opt['datapath'], 'wikipedia/full/wiki_full_extracted')

    def get_num_samples(self, opt) -> int:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt['datatype']
        if 'train' in datatype:
            return 12774693, 12774693
        elif 'valid' in datatype:
            return 7410, 7410
        else:
            return 7441, 7441

    def get_fold_chunks(self, opt) -> List[int]:  # type: ignore
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        datatype = opt['datatype']
        all_chunk_idxs = list(self.chunk_idx_to_file.keys())
        if DEBUG:
            print(f'Total chunks: {len(all_chunk_idxs)}')
        if 'train' in datatype:
            return all_chunk_idxs[:-10]
        elif 'valid' in datatype:
            return all_chunk_idxs[-10:-5]
        else:
            return all_chunk_idxs[-5:]

    def load_from_chunk(self, chunk_idx: int):
        """
        [Abstract] Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        output = []
        chunk_path = self.chunk_idx_to_file[chunk_idx]

        extra_data = []
        with open(chunk_path) as wf:
            for article_json in wf:
                article = json.loads(article_json)
                title = article['title']
                text = article['text']

                title = title.split(' (')[0]
                is_person = check_if_person(title)
                if not is_person:
                    continue

                gender = get_gender(text)

                label = f'ABOUT:{gender}'
                for par in text.split('\n'):
                    if par:
                        output.append((par, title, label, gender, 'about'))
                        self.counts[gender] += 1

                        if self.add_unknown_classes:
                            extra_data.append(
                                (
                                    par,
                                    title,
                                    f'SELF:{gend_utils.UNKNOWN}',
                                    gender,
                                    'self',
                                )
                            )
                            extra_data.append(
                                (
                                    par,
                                    title,
                                    f'PARTNER:{gend_utils.NEUTRAL}',
                                    gender,
                                    'partner',
                                )
                            )

        if len(extra_data) > 0:
            # possibly sample unknown classes
            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                to_samp = int(sample_rate * len(extra_data))
                sampled = random.sample(extra_data, to_samp)
                output += sampled
            else:
                output += extra_data

        if DEBUG:
            print('\n\nGender count update:')
            for k, v in self.counts.items():
                print(f'{k}: {v}')

        if (self.is_train and self.opt['balance']) or (
            self.is_valid and self.opt['balance_valid']
        ):
            exclude_lst = [
                f'ABOUT:{gend_utils.NONBINARY}',
                f'SELF:{gend_utils.UNKNOWN}',
                f'PARTNER:{gend_utils.NEUTRAL}',
            ]  # not enough of each of these examples to balance
            output = gend_utils.balance_data(output, key=2, exclude_labels=exclude_lst)

        if len(output) == 0:
            warn_once(f'CHUNK {chunk_idx} is empty')

        return output

    def _mask_gendered_words(self, text):
        return gend_utils.mask_gendered_words(text, self.gendered_list)

    def create_message(self, queue_output, entry_idx=0) -> 'Message':
        """
        [Abstract] Given the tuple output of the queue, return an act.
        """
        par, title, lbl, gender, class_type = queue_output
        if self.class_task == 'all':
            if class_type == 'self':
                labels = gend_utils.UNKNOWN_LABELS['self']  # Not True neutral
            else:
                labels = [lbl]
        elif self.class_task == 'single':
            labels = [gender]

        if self.mask_gendered_words:
            par = self._mask_gendered_words(par)

        return Message(
            {
                'text': par,
                'name': title,
                'labels': labels,
                'label_candidates': self.label_candidates[class_type],
                'episode_done': True,
                'id': 'Wikipedia Gender',
            }
        )

    def share(self):
        shared = super().share()
        shared['chunk_idx_to_file'] = self.chunk_idx_to_file
        return shared


def write_gender_to_file(open_file, loaded_data):
    prev = None
    for _, title, _, gender, _ in loaded_data:
        if title != prev:
            open_file.write(f'{title}\t{gender}\n')
            prev = title
