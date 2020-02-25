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
        parser = argparser.add_argument_group('ByteLevelBPE Arguments')
        parser.add_argument(
            '--bpe-vocab', type=str, help='path to pre-trained tokenizer vocab'
        )
        parser.add_argument(
            '--bpe-merge', type=str, help='path to pre-trained tokenizer merge'
        )
        parser.add_argument(
            '--bpe-add-prefix-space',
            type='bool',
            hidden=True,
            help='add prefix space before encoding',
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                'Please install HuggingFace tokenizer with: pip install tokenizers'
            )

        if 'bpe_vocab' not in opt:
            raise ValueError('--bpe-vocab is required for loading pretrained tokenizer')
        if 'bpe_merge' not in opt:
            raise ValueError('--bpe-merge is required for loading pretrained tokenizer')

        self.vocab_path = opt['bpe_vocab']
        self.merge_path = opt['bpe_merge']
        self.add_prefix_space = opt.get('bpe_add_prefix_space', True)
        self.tokenizer = ByteLevelBPETokenizer(
            self.vocab_path, self.merge_path, self.add_prefix_space
        )

    def encode(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens

    def decode(self, x: List[str]) -> str:
        return self.tokenizer.decode(self.tokenizer.token_to_id(c) for c in x)
