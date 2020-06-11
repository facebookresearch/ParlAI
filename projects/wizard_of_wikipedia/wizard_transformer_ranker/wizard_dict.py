#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent

from collections import defaultdict
import copy
import os

import re

RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)


class WizardDictAgent(DictionaryAgent):
    def __init__(self, opt, shared=None):
        # initialize fields
        self.opt = copy.deepcopy(opt)
        self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)
        self.null_token = '__PAD__'
        self.end_token = '__SOC__'
        self.unk_token = '__UNK__'
        self.start_token = '__SOC__'
        self.tokenizer = opt.get('dict_tokenizer', 'whitespace')
        self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        self.maxtokens = opt.get('dict_maxtokens', DictionaryAgent.default_maxtokens)
        self.textfields = opt.get(
            'dict_textfields', DictionaryAgent.default_textfields
        ).split(",")

        if shared:
            self.freq = shared.get('freq', {})
            self.tok2ind = shared.get('tok2ind', {})
            self.ind2tok = shared.get('ind2tok', {})
        else:
            self.freq = defaultdict(int)
            self.tok2ind = {}
            self.ind2tok = {}

            if opt.get('dict_file') and os.path.isfile(opt['dict_file']):
                # load pre-existing dictionary
                self.load(opt['dict_file'])
            elif opt.get('dict_initpath'):
                # load seed dictionary
                opt['dict_initpath'] = modelzoo_path(
                    opt.get('datapath'), opt['dict_initpath']
                )
                self.load(opt['dict_initpath'])

            self.add_token(self.null_token)
            self.add_token(self.start_token)
            self.add_token(self.end_token)
            self.add_token(self.unk_token)

        if not shared:
            if opt.get('dict_file'):
                self.save_path = opt['dict_file']

        # cache unk token for later
        self._unk_token_idx = self.tok2ind.get(self.unk_token)

    def tokenize(self, text, building=False):
        """
        Returns a sequence of tokens from the iterable.
        """
        if self.lower:
            text = text.lower()

        if self.tokenizer == 're':
            return self.re_tokenize(text)

        elif self.tokenizer == 'whitespace':
            return text.split(' ')

        word_tokens = (
            text.replace('.', ' . ')
            .replace('. . .', '...')
            .replace(',', ' , ')
            .replace(';', ' ; ')
            .replace(':', ' : ')
            .replace('!', ' ! ')
            .replace('?', ' ? ')
            .replace('  ', ' ')
            .replace('  ', ' ')
            .strip()
            .split(" ")
        )

        return word_tokens

    def re_tokenize(self, text):
        """
        This splits along whitespace and punctuation and keeps the newline as a token in
        the returned list.
        """
        return RETOK.findall(text)
