#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.dict import DictionaryAgent

try:
    from transformers import GPT2Tokenizer
except ImportError as e:
    raise e(
        'Need to install Hugging Face transformers repository. '
        'Try `pip install transformers`.'
    )
from abc import ABC, abstractmethod


class HuggingFaceDictionaryAgent(DictionaryAgent, ABC):
    """
    Use Hugging Face tokenizers.
    """

    def __init__(self, opt):
        super().__init__(opt)
        # initialize from vocab path
        self.tokenizer = self.get_tokenizer(opt)
        self.override_special_tokens(opt)

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

    def txt2vec(self, text, vec_type=list):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id

    def vec2txt(self, vec):
        return self.tokenizer.decode(vec, clean_up_tokenization_spaces=True)

    def act(self):
        """
        Dummy override.
        """
        return {}


class Gpt2DictionaryAgent(HuggingFaceDictionaryAgent):
    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        return GPT2Tokenizer.from_pretrained('gpt2')

    def override_special_tokens(self, opt):
        self.start_idx = -1  # hack, we end up removing this anyway
        self.end_token = self.tokenizer.eos_token  # "<|endoftext|>"
        self.end_idx = self.tokenizer.convert_tokens_to_ids([self.end_token])[0]
        self.pad_idx = 0
        # set tok2ind for special tokens
        self.tok2ind[self.end_token] = self.end_idx
        # set ind2tok for special tokens
        self.ind2tok[self.end_idx] = self.end_token
