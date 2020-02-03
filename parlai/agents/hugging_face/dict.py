#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.dict import DictionaryAgent

try:
    from transformers import GPT2Tokenizer
except ImportError:
    raise ImportError(
        'Need to install Hugging Face transformers repository. '
        'Try `pip install transformers`.'
    )
from abc import ABC, abstractmethod

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
}

NO_OP = "x"


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
        model_sz = opt['gpt2_size']
        fle_key = 'gpt2' if model_sz == 'small' else f'gpt2-{model_sz}'
        return GPT2Tokenizer.from_pretrained(fle_key)

    def _define_special_tokens(self, opt):
        if opt['add_special_tokens']:
            # Add addtional start/end/pad tokens
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
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
        self.start_idx = self.tokenizer.convert_tokens_to_ids([self.start_token])[0]
        self.end_idx = self.tokenizer.convert_tokens_to_ids([self.end_token])[0]
        self.null_idx = self.tokenizer.convert_tokens_to_ids([self.null_token])[0]
        # set tok2ind for special tokens
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.null_token] = self.null_idx
        # set ind2tok for special tokens
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.null_idx] = self.null_token
