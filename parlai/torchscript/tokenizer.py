#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch.jit


def is_alnum_or_underscore(ch: str):
    """
    A helper function for checking if a character is alphanumeric or underscore.
    """
    return ch.isalnum() or ch == '_'


@torch.jit.script
class ScriptableGpt2BpeHelper(object):
    """
    Version of parlai.utils.bpe.Gpt2BpeHelper that can be TorchScripted.
    """

    @classmethod
    def findall(cls, text: str) -> List[str]:
        """
        Split tokens in a manner that replicates parlai.utils.bpe.Gpt2BpeHelper.
        """
        contraction_endings = ["s", "t", "re", "ve", "m", "ll", "d"]

        tokens: List[str] = []
        idx = 0
        num_passes = 0
        while idx < len(text):
            num_passes += 1
            if num_passes > 10000:
                raise RuntimeError(
                    "*** Infinite loop in ScriptableGpt2BpeHelper.findall()! ***"
                )
            if text[idx] == "'":
                # Capture contradiction suffixes
                captured_suffix = False
                for ending in contraction_endings:
                    if text[idx + 1 : idx + 1 + len(ending)] == ending:
                        tokens.append("'" + ending)
                        idx += 1 + len(ending)
                        captured_suffix = True
                        break
                if captured_suffix:
                    continue
            if not text[idx].isspace() or (
                text[idx] == " " and idx + 1 < len(text) and not text[idx + 1].isspace()
            ):
                # Capture runs of one type of character
                if text[idx] == " ":
                    last_matching_idx = idx + 1
                else:
                    last_matching_idx = idx
                if text[last_matching_idx].isalpha():
                    while (
                        last_matching_idx + 1 < len(text)
                        and text[last_matching_idx + 1].isalpha()
                    ):
                        last_matching_idx += 1
                elif text[last_matching_idx].isnumeric():
                    while (
                        last_matching_idx + 1 < len(text)
                        and text[last_matching_idx + 1].isnumeric()
                    ):
                        last_matching_idx += 1
                else:
                    while (
                        last_matching_idx + 1 < len(text)
                        and not text[last_matching_idx + 1].isspace()
                        and not text[last_matching_idx + 1].isalpha()
                        and not text[last_matching_idx + 1].isnumeric()
                    ):
                        last_matching_idx += 1
                tokens.append(text[idx : last_matching_idx + 1])
                idx = last_matching_idx + 1
                continue
            if idx + 1 < len(text) and text[idx + 1].isspace():
                # Capture runs of space characters up until just before the final one
                last_space_idx = idx + 1
                while (
                    last_space_idx + 1 < len(text)
                    and text[last_space_idx + 1].isspace()
                ):
                    last_space_idx += 1
                if last_space_idx + 1 == len(text):
                    # Include the last char, which is a space char
                    tokens.append(text[idx : last_space_idx + 1])
                    idx = last_space_idx + 1
                else:
                    tokens.append(text[idx:last_space_idx])
                    idx = last_space_idx
                continue

            # Capture runs of space characters
            last_space_idx = idx
            while last_space_idx + 1 < len(text) and text[last_space_idx + 1].isspace():
                last_space_idx += 1
            tokens.append(text[idx : last_space_idx + 1])
            idx = last_space_idx + 1
        return tokens

    def __init__(
        self,
        add_prefix_space: bool,
        encoder: Dict[str, str],
        byte_encoder: Dict[int, str],
        fused_key_bpe_ranks: Dict[str, float],
        special_tokens: List[str],
    ):

        self.add_prefix_space = add_prefix_space

        self.encoder = encoder
        self.decoder: Dict[str, str] = {}
        if self.encoder is not None:
            for k, v in self.encoder.items():
                self.decoder[v] = k

        self.byte_encoder = byte_encoder
        self.byte_decoder: Dict[str, int] = {}
        if self.byte_encoder is not None:
            for k, v in self.byte_encoder.items():
                self.byte_decoder[v] = k

        self.bpe_ranks = fused_key_bpe_ranks

        # special tokens
        self._special_tokens: Dict[str, int] = {}
        for st in special_tokens:
            self._special_tokens[st] = 1

    def encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Checks for add_prefix_space; handles accordingly.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        if self.add_prefix_space:
            text = f" {text}"

        # constants for readability
        FINAL = 1
        SPLITABLE = 0
        pieces: List[Tuple[str, int]] = [(text, SPLITABLE)]

        for special_token in self._special_tokens.keys():
            i = 0
            while i < len(pieces):
                subtext, status = pieces[i]
                if status == FINAL:
                    i += 1
                    continue
                split = subtext.split(special_token)
                if len(split) > 1:
                    # special token detected, replace the chunk with small subchunks
                    # split by the special token
                    pieces.pop(i)
                    for j, piece in enumerate(split):
                        if j > 0:
                            # add the special token as a delimiter
                            pieces.insert(i + (2 * j) - 1, (special_token, FINAL))
                        pieces.insert(i + (2 * j), (piece, SPLITABLE))
                else:
                    i += 1

        output: List[str] = []
        for piece, state in pieces:
            if state is FINAL:
                output.append(piece)
            else:
                output += self.helper_encode(piece)
        text = "".join(output)

        return output

    def get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        """
        Return set of symbol pairs in a word.

        Word is represented as list of symbols (symbols being variable-length strings).

        :param word:
            word to symbolize

        :return pairs:
            set of tuples of symbols
        """
        pairs: List[Tuple[str, str]] = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, word: List[str]) -> List[str]:
        """
        Convert token to BPE.

        :param word:
            list of tokens token to convert

        :return bpe_encoding:
            string bpe encoding
        """
        pairs = self.get_pairs(word)

        if len(pairs) == 0:
            return word

        while True:
            min_rank = self.bpe_ranks.get("\n".join(pairs[0]), float("inf"))
            bigram = pairs[0]
            for pair in pairs[1:]:
                current_rank = self.bpe_ranks.get("\n".join(pair), float("inf"))
                if current_rank < min_rank:
                    min_rank = current_rank
                    bigram = pair
            if "\n".join(bigram) not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                found = False
                for j in range(i, len(word)):
                    if word[j] == first:
                        new_word.extend(word[i:j])
                        i = j
                        found = True
                        break
                if not found:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word.copy()
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        return word

    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        bpe_tokens: List[str] = []
        for token in self.findall(text):
            byte_encoded: List[str] = []
            for b in token:
                byte_encoded.append(self.byte_encoder[ord(b)])
            encoded: List[str] = []
            for bpe_token in self.bpe(byte_encoded):
                encoded.append(self.encoder[bpe_token])
            bpe_tokens.extend(encoded)
        return bpe_tokens

    def decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into a text string.

        :param tokens:
            list of tokens

        :return text:
            decoded text
        """
        output: List[str] = []
        accum: List[str] = []
        for token in tokens:
            if token in self._special_tokens:
                if len(accum) > 0:
                    output.append(self.helper_decode(accum))
                    accum.clear()
                output.append(token)
            else:
                accum.append(token)
        if len(accum) > 0:
            output.append(self.helper_decode(accum))

        text = "".join(output)
        if self.add_prefix_space:
            assert text.startswith(" ")
            text = text.lstrip(" ")
        return text

    def helper_decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens

        :return:
            decoded text
        """
        chars: List[str] = []
        for token in tokens:
            decoded_token = self.decoder[token]
            token_chars = self.utf8_chars(decoded_token)
            for char in token_chars:
                if not torch.jit.is_scripting():
                    # We iterate over "char", which is supposed to be a single
                    # character, because the TorchScripted version of the code
                    # correctly splits a string into single characters in
                    # self.utf8_chars() but the non-TorchScripted version doesn't
                    chars.extend(list(char))
                else:
                    chars.append(char)
        decoded_chars: List[str] = []
        for char in chars:
            decoded_chars.append(chr(self.byte_decoder[char]))
        return "".join(decoded_chars)

    def utf8_chars(self, s: str) -> List[str]:
        """
        An implementation of UTF8 character iteration in TorchScript. There are no
        bitwise operations in torchscript, so we compare directly to integer values.
        There isn't a lot of validation, for instance if you pass in an improperly
        encoded string with an out-of-place continuation byte, or with a non-left-to-
        right byte order, you'll get unexpected results and likely throw. Torch itself
        takes in unicode strings and encodes them as UTF8, so that should be actively
        hard to do.

        The logic is simple: looking at the current start-of-character byte.
        If its high bit is 0, it's a 1-byte character. Otherwise, the number of
        bytes is the number of leading 1s in its binary representation, so
        find that number by comparing it directly to ints with the appropriate
        representation, then append that many bytes as a character and move past
        them to the next start byte.

        From pytext.torchscript.utils.
        """
        chars: List[str] = []
        i = 0
        while i < len(s):
            byte = ord(s[i])
            if byte < 0b10000000:
                chars.append(s[i])
                i += 1
            else:
                if byte < 0b11100000:
                    num_bytes = 2
                elif byte < 0b11110000:
                    num_bytes = 3
                elif byte < 0b11111000:
                    num_bytes = 4
                elif byte < 0b11111100:
                    num_bytes = 5
                elif byte < 0b11111110:
                    num_bytes = 6
                elif byte < 0b11111111:
                    num_bytes = 7
                else:
                    num_bytes = 8
                chars.append(s[i : i + num_bytes])
                i += num_bytes
        return chars


@torch.jit.script
class ScriptableSubwordBpeHelper(object):
    """
    Version of parlai.utils.bpe.SubwordBPEHelper that can be TorchScripted.
    """

    @classmethod
    def findall(cls, text: str) -> List[str]:
        """
        Split tokens in a manner that replicates parlai.utils.bpe.Gpt2BpeHelper.
        """
        tokens: List[str] = []
        idx = 0
        num_passes = 0
        while idx < len(text):
            num_passes += 1
            if num_passes > 10000:
                raise RuntimeError(
                    "*** Infinite loop in ScriptableSubwordBpeHelper.findall()! ***"
                )

            if not text[idx].isspace():
                last_matching_idx = idx
                if is_alnum_or_underscore(text[last_matching_idx]):
                    while last_matching_idx + 1 < len(text) and (
                        text[last_matching_idx + 1].isalnum()
                        or text[last_matching_idx + 1] == '_'
                    ):
                        last_matching_idx += 1
                tokens.append(text[idx : last_matching_idx + 1])
                idx = last_matching_idx + 1
                continue

            # Capture runs of space characters
            last_space_idx = idx
            while last_space_idx + 1 < len(text) and text[last_space_idx + 1].isspace():
                last_space_idx += 1
            idx = last_space_idx + 1
        return tokens

    def __init__(
        self,
        add_prefix_space: bool,
        version: Tuple[int, int],
        bpe_codes: Dict[str, float],
        separator: str,
    ):

        self.add_prefix_space = add_prefix_space
        self.version = version
        self.bpe_codes = bpe_codes
        self.separator = separator

    def encode(self, text: str) -> List[str]:
        if self.add_prefix_space:
            text = f" {text}"
        return self.helper_encode(text)

    def get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        """
        Return set of symbol pairs in a word.

        Word is represented as list of symbols (symbols being variable-length strings).

        :param word:
            word to symbolize

        :return pairs:
            set of tuples of symbols
        """
        pairs: List[Tuple[str, str]] = []
        prev_char = word[0]

        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char

        return pairs

    def bpe(self, orig: str) -> List[str]:
        """
        Convert token to BPE.

        :param word:
            token to convert

        :return bpe_encoding:
            string bpe encoding
        """
        if self.version == (0, 2):
            word = list(orig[:-1]) + [orig[-1] + '</w>']
        else:
            raise NotImplementedError

        pairs = self.get_pairs(word)
        if len(pairs) == 0:
            return [orig]

        while True:
            min_rank = self.bpe_codes.get("\n".join(pairs[0]), float("inf"))
            bigram = pairs[0]
            for pair in pairs[1:]:
                current_rank = self.bpe_codes.get("\n".join(pair), float("inf"))
                if current_rank < min_rank:
                    min_rank = current_rank
                    bigram = pair
            if "\n".join(bigram) not in self.bpe_codes:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                found = False
                for j in range(i, len(word)):
                    if word[j] == first:
                        new_word.extend(word[i:j])
                        i = j
                        found = True
                        break
                if not found:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word.copy()
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        # don't print end-of-word symbols
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + [word[-1].replace('</w>', '')]
        return word

    def segment_tokens(self, tokens: List[str]) -> List[str]:
        """
        segment a sequence of tokens with BPE encoding.
        """
        output: List[str] = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue

            new_word: List[str] = []
            for out in self.bpe(word):
                new_word.append(out)

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def helper_encode(self, text: str) -> List[str]:
        text = text.replace('\n', ' __newln__ ')
        tokens = self.findall(text)
        return self.segment_tokens(tokens)

    def decode(self, tokens: List[str]) -> str:
        output: List[str] = []
        accum: List[str] = []
        for token in tokens:
            accum.append(token)
        if len(accum) > 0:
            output.append(self.helper_decode(accum, ' '))

        text = "".join(output)
        if self.add_prefix_space:
            assert text.startswith(" ")
            text = text.lstrip(" ")
        return text

    def helper_decode(self, tokens: List[str], delimiter: str = ' ') -> str:
        text = delimiter.join(tokens)
        text = text.replace('@@ ', '')
        # It's also possible that we get a BPE encoding on the end of the word
        if text.endswith('@@'):
            text = text[:-2]
        text = text.replace('__newln__', '\n')
        return text


@torch.jit.script
class ScriptableDictionaryAgent:
    """
    Builds and/or loads a dictionary.

    All code is TorchScriptable.
    """

    def __init__(
        self,
        null_token: str,
        end_token: str,
        unk_token: str,
        start_token: str,
        freq: Dict[str, int],
        tok2ind: Dict[str, int],
        ind2tok: Dict[int, str],
        bpe_add_prefix_space: bool,
        bpe_encoder: Dict[str, str],
        bpe_byte_encoder: Dict[int, str],
        fused_key_bpe_ranks: Dict[str, float],
        special_tokens: List[str],
        subword_bpe_version: Tuple[int, int],
        fused_bpe_codes: Dict[str, float],
        subword_bpe_separator: str,
    ):

        self.null_token = null_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.start_token = start_token

        self.freq = freq
        self.tok2ind = tok2ind
        self.ind2tok = ind2tok

        # cache unk token for later
        self._unk_token_idx = self.tok2ind[self.unk_token]

        # Initialize tokenizer
        self.subword_bpe = ScriptableSubwordBpeHelper(
            add_prefix_space=bpe_add_prefix_space,
            version=subword_bpe_version,
            bpe_codes=fused_bpe_codes,
            separator=subword_bpe_separator,
        )

        self.gpt2_bpe = ScriptableGpt2BpeHelper(
            add_prefix_space=bpe_add_prefix_space,
            encoder=bpe_encoder,
            byte_encoder=bpe_byte_encoder,
            fused_key_bpe_ranks=fused_key_bpe_ranks,
            special_tokens=special_tokens,
        )

    def _word_lookup(self, key: str) -> int:
        """
        Return index from token, or unk_token's index, or None.
        """
        if key in self.tok2ind:
            return self.tok2ind[key]
        else:
            return self._unk_token_idx

    def _index_lookup(self, key: int) -> str:
        """
        Return token from index, or unk_token.
        """
        if key in self.ind2tok:
            return self.ind2tok[key]
        else:
            return self.unk_token

    def gpt2_tokenize(self, text: str):
        """
        Tokenize using Gpt2 BPE tokenizer.
        """
        return self.gpt2_bpe.encode(text)

    def bpe_tokenize(self, text: str) -> List[str]:
        """
        Return a sequence of BPE-tokens from the text.
        """
        return self.subword_bpe.encode(text)

    def txt2vec(self, text: str, dict_tokenizer: str) -> List[int]:
        """
        Convert a string to a vector (list of ints).

        First runs a sentence tokenizer, then a word tokenizer.
        """
        itr: List[int] = []
        if dict_tokenizer == 'gpt2' or dict_tokenizer == 'slow_bytelevel_bpe':
            for token in self.gpt2_tokenize(str(text)):
                itr.append(self._word_lookup(token))
        elif dict_tokenizer == 'bpe':
            for token in self.bpe_tokenize(str(text)):
                itr.append(self._word_lookup(token))
        else:
            raise NotImplementedError
        return itr

    def vec2txt(self, vector: List[int], dict_tokenizer: str) -> str:
        """
        Convert a vector of IDs to a string.

        Converts a vector (iterable of ints) into a string, with each token separated by
        the delimiter (default ``' '``).
        """
        tokens = [self._index_lookup(idx) for idx in vector]
        if dict_tokenizer == 'gpt2' or dict_tokenizer == 'slow_bytelevel_bpe':
            text = self.gpt2_bpe.decode(tokens)
        elif dict_tokenizer == 'bpe':
            text = self.subword_bpe.decode(tokens)
        else:
            raise NotImplementedError
        return text
