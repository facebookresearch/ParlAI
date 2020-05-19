#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Taskmaster-2 implementation for ParlAI.

No official train/valid/test splits are available as of 2020-05-18, so we make
our own splits.
"""

import os
import pandas as pd
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once

import parlai.tasks.taskmaster2.build as build_

SECTIONS = [
    'flights',
    'food-ordering',
    'hotels',
    'movies',
    'restaurant-search',
    'sports',
]


class _Abstract(DialogTeacher):
    """
    Abstract data loader.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        return argparser

    def __init__(self, opt: Opt, shared=None):
        self.fold = opt['datatype'].split(':')[0]
        opt['datafile'] = self.fold
        self.dpath = os.path.join(opt['datapath'], 'taskmaster-2')
        if shared is None:
            warn_once(
                "Taskmaster2 is a beta dataset, and format may significantly change."
            )
            build_.build(opt)
        super().__init__(opt, shared)

    def _h(self, x):
        h = abs(hash(x)) % 10
        if h == 0:
            return 'valid'
        elif h == 1:
            return 'test'
        else:
            return 'train'

    def _label_fold(self, chunks):
        return chunks.conversation_id.apply(self._h)

    def _load_data(self, fold):
        chunks = []
        for section in SECTIONS:
            subset = pd.read_json(os.path.join(self.dpath, section + '.json'))
            chunks.append(subset)
        chunks = pd.concat(chunks, axis=0)
        chunks['fold'] = self._label_fold(chunks)
        chunks = chunks[chunks.fold == fold].reset_index()
        chunks = chunks.sample(frac=1.0, random_state=42)
        return chunks

    def _segments2text(self, segments):
        output = []
        for segment in segments:
            val = segment['text']
            for anno_ in segment['annotations']:
                anno = anno_['name']
                output.append(f'{anno} = {val}')
        return " ; ".join(output)

    def setup_data(self, fold):
        chunks = self._load_data(fold)
        for _, row in chunks.iterrows():
            last = None
            first = True
            utterances = row['utterances'][:]
            if (
                len(utterances) >= 3
                and utterances[0]['speaker'] == 'USER'
                and utterances[1]['speaker'] == 'ASSISTANT'
                and utterances[2]['speaker'] == 'ASSISTANT'
                and "help you?" in utterances[1]['text']
            ):
                # skip this one
                utterances.pop(1)
            while utterances:
                utt = utterances.pop(0)
                seg = self._segments2text(utt.get('segments', []))
                if utt['speaker'] == 'USER':
                    yield (utt['text'], 'APICALL: ' + seg), first
                    first = False
                elif utt['speaker'] == 'ASSISTANT':
                    yield ('APIRESP: ' + seg, utt['text']), first
                    first = False
                last = utt['speaker']


class DefaultTeacher(_Abstract):
    pass
