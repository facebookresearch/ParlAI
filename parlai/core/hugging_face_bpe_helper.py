# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from parlai.core.opt import Opt


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
        if opt.get('bpe_vocab', None) is None:
            raise ValueError('--bpe-vocab is required for loading pretrained tokenizer')
        if opt.get('bpe_merge', None) is None:
            raise ValueError('--bpe-merge is required for loading pretrained tokenizer')
        self.vocab_path = opt.get('bpe_vocab')
        self.merge_path = opt.get('bpe_merge')
        try:
            from tokenizers import ByteLevelBPETokenizer

            self.byte_Level_Bpe = ByteLevelBPETokenizer(
                self.vocab_path, self.merge_path
            )
        except ImportError:
            raise ImportError(
                'Please install huggingface tokenizer with: pip install tokenizers'
            )

    def encode(self, text: str) -> List[str]:
        return self.byte_Level_Bpe.encode(text).tokens

    def decode(self, x: List[int]) -> str:
        return self.byte_Level_Bpe.decode(x)
