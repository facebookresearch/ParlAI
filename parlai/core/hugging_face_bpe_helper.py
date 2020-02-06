#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.opt import Opt
from typing import List


class HuggingFaceBpeHelper(object):
    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('HuggingFaceBpeHelper Arguments')
        parser.add_argument(
            '--bpe-vocab', type=str, help='path to pre-trained tokenizer vocab'
        )
        parser.add_argument(
            '--bpe-merge', type=str, help='path to pre-trained tokenizer merge'
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                'Please install huggingface tokenizer with: pip install tokenizers'
            )

        if opt.get('bpe_vocab', None) is None:
            raise ValueError('--bpe-vocab is required for loading pretrained tokenizer')
        if opt.get('bpe_merge', None) is None:
            raise ValueError('--bpe-merge is required for loading pretrained tokenizer')
        self.vocab_path = opt.get('bpe_vocab')
        self.merge_path = opt.get('bpe_merge')
        self.tokenizer = ByteLevelBPETokenizer(self.vocab_path, self.merge_path)

    def encode(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens

    def decode(self, x: List[str]) -> str:
        return self.tokenizer.decode(self.tokenizer.token_to_id(c) for c in x)
