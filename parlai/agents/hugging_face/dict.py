#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import List

from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.utils.io import PathManager


try:
    from transformers import GPT2Tokenizer, T5TokenizerFast
except ImportError:
    raise ImportError(
        "Need to install Hugging Face transformers repository. "
        "Try `pip install transformers`."
    )

SPECIAL_TOKENS = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}

NO_OP = "x"


class HuggingFaceDictionaryAgent(DictionaryAgent, ABC):
    """
    Use Hugging Face tokenizers.
    """

    def __init__(self, opt: Opt, shared=None):
        if not shared:
            self.hf_tokenizer = self.get_tokenizer(opt)
            self.tok2ind = self.hf_tokenizer.get_vocab()
            self.ind2tok = {v: k for k, v in self.tok2ind.items()}
        else:
            self.hf_tokenizer = shared['hf_tokenizer']
            self.tok2ind = shared['tok2ind']
            self.ind2tok = shared['ind2tok']

        self.freq = defaultdict(int)
        for tok in self.tok2ind:
            self.freq[tok] = 1
        self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)

        self._unk_token_idx = self.hf_tokenizer.unk_token_id
        self.override_special_tokens(opt)

        self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        self.tokenizer = 'hf'
        self.opt = opt
        self.max_length = (
            self.opt.get('text_truncate') or self.hf_tokenizer.model_max_length
        )

    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.

        Overrides DictionaryAgent.is_prebuilt
        """
        return True

    @abstractmethod
    def get_tokenizer(self, opt):
        """
        Instantiate the HuggingFace tokenizer for your model.
        """
        pass

    @abstractmethod
    def override_special_tokens(opt):
        """
        Override the special tokens for your tokenizer.
        """
        pass

    @abstractproperty
    def add_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """

    @abstractproperty
    def skip_decode_special_tokens(self) -> bool:
        """
        Whether to skip special tokens when converting tokens to text.
        """

    def share(self):
        shared = super().share()
        shared['hf_tokenizer'] = self.hf_tokenizer
        shared['ind2tok'] = self.ind2tok
        shared['tok2ind'] = self.tok2ind
        return shared

    def format_text(self, text: str) -> str:
        """
        Format text prior to encoding with tokenizer.
        """
        return text

    def txt2vec(self, text, vec_type=list):
        return self.hf_tokenizer.encode(
            self.format_text(text),
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            pad_to_max_length=False,
            truncation='longest_first',
        )

    def vec2txt(self, vec, **kwargs):
        return self.hf_tokenizer.decode(
            vec, skip_special_tokens=self.skip_decode_special_tokens, **kwargs
        )

    def act(self):
        return {}


class Gpt2DictionaryAgent(HuggingFaceDictionaryAgent):
    @property
    def add_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """
        return True

    @property
    def skip_decode_special_tokens(self) -> bool:
        """
        Whether to skip special tokens when converting tokens to text.
        """
        return False

    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        if opt.get("model_name"):
            fle_key = opt["model_name"]
        else:
            model_sz = opt["gpt2_size"]
            if model_sz == "small":
                model_key = "gpt2"
            elif model_sz == "distilgpt2":
                model_key = "distilgpt2"
            else:
                model_key = f"gpt2-{model_sz}"
            # check if datapath has the files that hugging face code looks for
            hf_dir = os.path.join(opt["datapath"], "hf", model_key)
            if all(
                PathManager.exists(os.path.join(hf_dir, file_name))
                for file_name in ["merges.txt", "vocab.json"]
            ):
                fle_key = PathManager.get_local_path(hf_dir, recursive=True)

            else:
                fle_key = model_key
        return GPT2Tokenizer.from_pretrained(fle_key)

    def add_additional_special_tokens(self, additional_special_tokens: List[str]):
        """
        Add additional special tokens to the dictionary.
        """
        self.additional_special_tokens = additional_special_tokens
        self.hf_tokenizer.add_special_tokens(
            {'additional_special_tokens': additional_special_tokens}
        )
        for tok in self.additional_special_tokens:
            self.add_token(tok)

    def _define_special_tokens(self, opt):
        if opt["add_special_tokens"]:
            # Add additional start/end/pad tokens
            self.hf_tokenizer.add_special_tokens(SPECIAL_TOKENS)
            self.start_token = SPECIAL_TOKENS["bos_token"]
            self.end_token = SPECIAL_TOKENS["eos_token"]
            self.null_token = SPECIAL_TOKENS["pad_token"]
        else:
            # Only special token is end of text
            self.start_token = NO_OP  # hack, we cut off the start token
            self.end_token = "<|endoftext|>"
            self.null_token = "<|endoftext|>"

    def override_special_tokens(self, opt):
        # define special tokens
        self._define_special_tokens(opt)
        # now override
        self.start_idx = self.hf_tokenizer.convert_tokens_to_ids([self.start_token])[0]
        self.end_idx = self.hf_tokenizer.convert_tokens_to_ids([self.end_token])[0]
        self.null_idx = self.hf_tokenizer.convert_tokens_to_ids([self.null_token])[0]
        # set tok2ind for special tokens
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.null_token] = self.null_idx
        # set ind2tok for special tokens
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.null_idx] = self.null_token


class DialoGPTDictionaryAgent(Gpt2DictionaryAgent):
    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        model_sz = opt["gpt2_size"]
        fle_key = f"microsoft/DialoGPT-{model_sz}"
        return GPT2Tokenizer.from_pretrained(fle_key)


class T5DictionaryAgent(HuggingFaceDictionaryAgent):
    def get_tokenizer(self, opt):
        return T5TokenizerFast.from_pretrained(opt['t5_model_arch'], truncation=True)

    @property
    def add_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """
        return True

    @property
    def skip_decode_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """
        return True

    def override_special_tokens(self, opt):
        # now override
        self.start_token = self.hf_tokenizer.pad_token
        self.end_token = self.hf_tokenizer.eos_token
        self.null_token = self.hf_tokenizer.pad_token
        self.unk_token = self.hf_tokenizer.unk_token

        self._unk_token_idx = self.hf_tokenizer.unk_token_id

        self.start_idx = self[self.start_token]
        self.end_idx = self[self.end_token]
        self.null_idx = self[self.null_token]
