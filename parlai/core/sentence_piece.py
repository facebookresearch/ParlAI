#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

try:
    import sentencepiece as spm
except ImportError:
    raise ImportError("Need to install SentencePiece. Run `pip install sentencepiece`.")

from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.utils import logging
from parlai.utils.io import PathManager


class SentencePieceDictionaryAgent(DictionaryAgent):
    def __init__(self, opt: Opt, shared=None):
        if not shared:
            self.sp_model = self.load(opt['dict_file'])
            self.tok2ind = self.sp_model
            self.ind2tok = {}
            self.freq = defaultdict(int)
        else:
            self.sp_model = shared['sp_model']
            self.tok2ind = shared['tok2ind']
            self.ind2tok = shared['ind2tok']
            self.freq = shared['freq']

        # self.null_token = self.sp_model.id_to_piece(self.sp_model.pad_id())
        # self.unk_token = self.sp_model.id_to_piece(self.sp_model.unk_id())
        # self.start_token = self.sp_model.id_to_piece(self.sp_model.bos_id())
        # self.end_token = self.sp_model.id_to_piece(self.sp_model.eos_id())
        self.null_token = opt.get('dict_nulltoken', DictionaryAgent.default_null)
        self.end_token = opt.get('dict_endtoken', DictionaryAgent.default_end)
        self.unk_token = opt.get('dict_unktoken', DictionaryAgent.default_unk)
        self.start_token = opt.get('dict_starttoken', DictionaryAgent.default_start)

        self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)

        self._unk_token_idx = self.sp_model.unk_id()

        self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        self.tokenizer = 'sentence_piece'
        self.opt = opt
        self.max_length = self.opt.get('text_truncate')

    def share(self):
        shared = super().share()
        shared['sp_model'] = self.sp_model
        shared['tok2ind'] = self.tok2ind
        return shared

    def sentence_piece_tokenize(self, text):
        return self.sp_model.encode(text)

    def txt2vec(self, text, vec_type=list):
        return self.sp_model.encode(text)

    def vec2txt(self, vec, **kwargs):
        return self.sp_model.decode(vec)

    def act(self):
        """
        Override so new tokens aren't added to the dict.
        """
        return {}

    def load(self, filename: str):
        """
        Load pre-existing dictionary in 'token[<TAB>count]' format.

        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        logging.info(f'loading dictionary from {filename}')
        sp_model = spm.SentencePieceProcessor(model_file=filename)
        logging.info(f'vocab size = {len(sp_model)}')
        return sp_model

    def save(self, filename=None, append=False, sort=True):
        """
        Override save, since model should not be overwritten.
        """
        pass

    def is_prebuilt(self):
        return True

    def __contains__(self, key):
        if type(key) == int:
            return key >= 0 and key < self.sp_model.vocab_size()
        elif type(key) == str:
            return (
                key == self.unk_token
                or self.sp_model.piece_to_id(key) != self.sp_model.unk_id()
            )

    def __getitem__(self, key):
        if type(key) == str:
            return self.sp_model[key]
        if type(key) == int:
            return self.sp_model.id_to_piece(key)
