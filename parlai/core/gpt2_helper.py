#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Byte pair encoding utilities from GPT-2.

Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT
This is an implemtation from fairseq : https://github.com/pytorch/fairseq/blob/master/fairseq/data/encoders/gpt2_bpe_utils.py
Implemtation license: MIT
"""

from typing import List
from parlai.core.opt import Opt
from functools import lru_cache
import json
from .build_data import download, make_dir
import os

DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings. This means you need a large # of
    unicode characters in your vocab if you want to avoid UNKs. When you're at something
    like a 10B token dataset you end up needing around 5K for decent coverage. This is a
    signficant percentage of your normal, say, 32K bpe vocab. To avoid that, we want
    lookup tables between utf-8 bytes and unicode strings. And avoids mapping to
    whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Gpt2BpeHelper:
    def __init__(self, opt: Opt, errors='replace'):
        data_path = os.path.join(opt['datapath'], 'gpt2')
        vocab_path = os.path.join(data_path, 'vocab.bpe')
        json_path = os.path.join(data_path, 'encoder.json')
        if not os.path.isfile(vocab_path) or not os.path.isfile(json_path):
            make_dir(data_path)
            download(DEFAULT_VOCAB_BPE, data_path, 'vocab.bpe')
            download(DEFAULT_ENCODER_JSON, data_path, 'encoder.json')
        with open(vocab_path, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        with open(json_path, 'r') as f:
            self.encoder = json.load(f)
        for each_token in self.encoder.keys():
            new_token = ''.join(
                # escape nonprintable characters
                '\\' + hex(b).lstrip('0') if (b > 127 or b < 32) else chr(b)
                for b in each_token.encode('utf-8')
            )
            self.encoder[each_token] = new_token
        self.decoder = {v: k for k, v in self.encoder.items()}
        bpe_merges = [
            tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
        ]
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        try:
            import regex as re

            self.re = re
        except ImportError:
            raise ImportError('Please install regex with: pip install regex')

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @lru_cache(maxsize=10240)
    def bpe(self, token: str) -> str:
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        return ' '.join(word)

    def encode(self, text: str) -> List[str]:
        bpe_tokens: List[str] = []
        for token in self.re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')
            )
        return bpe_tokens

    def decode(self, tokens: List[str]) -> str:
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors
        )
        return text

    def list_tokens(self):
        return self.encoder.values()
