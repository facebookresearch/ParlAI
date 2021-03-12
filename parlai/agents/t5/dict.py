#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Wrapped HF Tokenizer as a ParlAI DictionaryAgent.
"""
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from transformers import T5TokenizerFast

from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt


class HFTokenizerDictionaryAgent(DictionaryAgent, ABC):
    """
    Handle Dict Agent responsibilities using a Tokenizer from HF.
    """

    def __init__(self, opt: Opt, shared=None):
        if not shared:
            self.hf_tokenizer = self.build_hf_tokenizer(opt)
            self.tok2ind = self.hf_tokenizer.get_vocab()
            self.ind2tok = {v: k for k, v in self.tok2ind.items()}
        else:
            self.hf_tokenizer = shared['hf_tokenizer']
            self.tok2ind = shared['tok2ind']
            self.ind2tok = shared['ind2tok']

        self.freq = defaultdict(int)
        self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)

        self.start_token = self.hf_tokenizer.cls_token
        self.end_token = self.hf_tokenizer.sep_token
        self.null_token = self.hf_tokenizer.pad_token
        self.unk_token = self.hf_tokenizer.unk_token

        self._unk_token_idx = self.hf_tokenizer.unk_token_id
        self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        self.tokenizer = 'bert'
        self.opt = opt
        self.max_length = (
            self.opt['text_truncate'] or self.hf_tokenizer.model_max_length
        )

    def is_prebuilt(self):
        return True

    @abstractmethod
    def build_hf_tokenizer(self, opt):
        """
        Return hf tokenizer.
        """

    @abstractmethod
    def format_text(self, text: str) -> str:
        """
        Format text prior to encoding with tokenizer.
        """

    @abstractproperty
    def add_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """

    def share(self):
        shared = super().share()
        shared['hf_tokenizer'] = self.hf_tokenizer
        shared['ind2tok'] = self.ind2tok
        shared['tok2ind'] = self.tok2ind
        return shared

    def __len__(self):
        if hasattr(self, 'hf_tokenizer'):
            return self.hf_tokenizer.vocab_size
        else:
            return super().__len__()

    def txt2vec(self, text, vec_type=list):
        return self.hf_tokenizer.encode(
            self.format_text(text),
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            pad_to_max_length=False,
            truncation='longest_first',
        )

    def vec2txt(self, vec, **kwargs):
        return self.hf_tokenizer.decode(vec, skip_special_tokens=True, **kwargs)

    def act(self):
        return {}


class T5TokenizerDictionaryAgent(HFTokenizerDictionaryAgent):
    """
    Handle Dict Agent responsibilities using a BERT Tokenizer from HF.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)

        self.start_token = self.hf_tokenizer.pad_token
        self.end_token = self.hf_tokenizer.eos_token
        self.null_token = self.hf_tokenizer.pad_token
        self.unk_token = self.hf_tokenizer.unk_token

        self._unk_token_idx = self.hf_tokenizer.unk_token_id

    def build_hf_tokenizer(self, opt):
        return T5TokenizerFast.from_pretrained(opt['t5_model_arch'], truncation=True)

    def format_text(self, text: str) -> str:
        """
        Format text prior to encoding with tokenizer.
        """
        return text

    @property
    def add_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """
        return True
