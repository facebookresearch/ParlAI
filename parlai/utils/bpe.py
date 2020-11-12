#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Byte pair encoding (BPE).

Lots of BPE things for ParlAI
"""
from abc import ABC, abstractmethod
from functools import lru_cache
import json
import os
import random
import re
from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import final

from parlai.core.build_data import download, make_dir
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.typing import TShared
from parlai.utils.io import PathManager
import parlai.utils.logging as logging

try:
    from subword_nmt import learn_bpe, apply_bpe

    # Don't explicitly throw the runtime error unless the user needs it
    SUBWORD_BPE_INSTALLED = True
except ImportError:
    SUBWORD_BPE_INSTALLED = False


def bpe_factory(opt: Opt, shared: TShared) -> 'BPEHelper':
    """
    BPE Helper Factory.

    Returns the appropriate BPE helper given the opt
    as well as available libraries.

    :param opt:
        options
    :param shared:
        shared dict

    :return BPEHelper:
        returns the appropriate BPEHelper object
    """
    from parlai.core.dict import DictionaryAgent

    tokenizer = opt.get('dict_tokenizer', DictionaryAgent.default_tok)

    bpe_helper: Optional[BPEHelper] = None

    if tokenizer == 'bytelevelbpe':
        # Attempt to instantiate HF tokenizer
        try:
            bpe_helper = HuggingFaceBpeHelper(opt, shared)
        except ImportError:
            if opt['dict_loaded']:
                warn_once(
                    ''
                    '\n\n--------------------------------------------------\n\n'
                    'WARNING: You have chosen to use Huggingface\'s tokenizer.\n'
                    'Please install HuggingFace tokenizer with: pip install tokenizers.\n'
                    'For now, defaulting to the GPT2Tokenizer.'
                    '\n\n--------------------------------------------------\n\n'
                )
                tokenizer = 'slow_bytelevel_bpe'
            else:
                raise ImportError(
                    'Please install HuggingFace tokenizer with: pip install tokenizers.\n'
                )
    if tokenizer == 'slow_bytelevel_bpe':
        bpe_helper = SlowBytelevelBPE(opt, shared)
    if tokenizer == 'gpt2':
        bpe_helper = Gpt2BpeHelper(opt, shared)
    if tokenizer == 'bpe':
        bpe_helper = SubwordBPEHelper(opt, shared)

    assert (
        bpe_helper is not None
    ), f"bpe_factory called with invalid tokenizer: {tokenizer}"

    return bpe_helper


class BPEHelper(ABC):
    """
    Abstract BPE Helper.

    BPE Helper subclasses must implement appropriate abstractmethods.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        """
        Subclasses _should_ override __init__ to initialize other things.
        """
        from parlai.core.dict import DictionaryAgent

        self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        self.maxtokens = opt.get('dict_maxtokens', DictionaryAgent.default_maxtokens)
        self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)

        self.opt = opt
        self.debug = opt.get('bpe_debug', False)
        self.add_prefix_space = opt.get('bpe_add_prefix_space', False)
        self._special_tokens: Dict[str, int] = {}
        self.bpe_dropout: Optional[float] = opt.get('bpe_dropout')
        self._bpe_dropout_enabled = False

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('BPEHelper Arguments')
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
        parser.add_argument(
            '--bpe-dropout',
            type=float,
            default=None,
            help='Use BPE dropout during training.',
        )
        return parser

    def enable_bpe_dropout(self, enabled: bool):
        """
        Used to toggle BPE dropout on (True) or off (False).
        """
        self._bpe_dropout_enabled = enabled

    @final
    def encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Checks for add_prefix_space; handles accordingly

        NOTE: DO NOT OVERRIDE

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        for special_token in self._special_tokens.keys():
            split = text.split(special_token)
            if len(split) > 1:
                output = []
                for i, piece in enumerate(split):
                    if i > 0:
                        output.append(special_token)
                    output += self.encode(piece)
                return output
        if self.add_prefix_space and not isinstance(self, HuggingFaceBpeHelper):
            text = f' {text}'
        return self.helper_encode(text)

    @abstractmethod
    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Subclasses should override this method for encoding.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """

    @final
    def decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str = ' '
    ) -> str:
        """
        Decode list of tokens into a text string.

        NOTE: DO NOT OVERRIDE

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        if self.debug:
            return delimiter.join(tokens)

        for i, token in enumerate(tokens):
            # note, HF ByteLevelBPE tokenizer handles special tokens itself in
            # a special way, so this will be skipped
            if token in self._special_tokens:
                # special token found. to the left, we've already cleared
                left = self.helper_decode(tokens[:i], token_ids[:i], delimiter)
                # token itself is easy to map to a string
                center = token
                # to the right, there may stil be special tokens
                right = self.decode(
                    tokens[min(len(token_ids), i + 1) :],
                    token_ids[min(len(token_ids), i + 1) :],
                    delimiter,
                )
                return left + center + right

        # no special tokens found, we can fall back
        text = self.helper_decode(tokens, token_ids, delimiter)
        if self.add_prefix_space:
            assert text.startswith(' ')
            text = text.lstrip(' ')
        return text

    @abstractmethod
    def helper_decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str
    ) -> str:
        """
        Decode list of tokens into text string.

        Subclasses should override this method for decoding.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """

    @abstractmethod
    def sync_with_dict(self, dict_agent):
        """
        Sync BPE Helper dictionary with dict_agent dict.

        :param dict_agent:
            agent with which we are syncing the dictionary
        """

    def add_special_tokens(self, dict_agent, special_tokens: List[str]):
        """
        Add special tokens to the tokenizer.

        These tokens are never split, and prioritized over the BPE tokenization.
        """
        # note, HF ByteLevelBPE tokenizer handles special tokens itself in
        # a special way, so this will be skipped
        for token in special_tokens:
            # exploiting dictionaries' insertion ordering to emulate ordered sets
            self._special_tokens[token] = 1

    def finalize(
        self, frequencies: Dict[str, int], num_symbols: int, minfreq: int
    ) -> bool:
        """
        Build the codecs.

        Default helpers are pre-trained and thus do not build their own codecs

        :param frequencies:
            dictionary of (token: frequency) pairs
        :param num_symbols:
            Number of BPE symbols. Recommend 30000-40000.  If <= 0, default
            30000 will be used.
        :param minfreq:
            Minimum frequency of a token before forced BPE decomposition. If <=
            0 will use subword-nmt default of 2.

        :return did_finalize:
            return whether codecs are finalized this call.
        """
        return False

    def copy_codecs_file(self, target_file: str):
        """
        Copy the codecs file to a new location.

        Default behavior is to do nothing.

        :param target_file:
            where to copy the codecs.
        """
        pass

    def should_sort(self) -> bool:
        """
        Return whether tokens should be sorted for this particular helper.

        DictionaryAgent sorts tokens upon saving; we don't generally want to sort with
        our pre-trained dictionaries, so default is False.
        """
        return False


###############
# Subword BPE #
###############


class SubwordBPEHelper(BPEHelper):
    """
    Helper class for performing BPE subword tokenization.

    For technical details, please refer to https://arxiv.org/abs/1508.07909.
    This class just wraps around the official subword-nmt repository.

    This API expects the user to call tokenize() (encode) onto the training data,
    then call finalize() to learn the encodings, and then iterate over the data
    in a second pass, calling tokenize() again to get processed output.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        """
        Initialize the BPE module.

        :param opt:
            options
        :param shared:
            shared dictionary
        """
        super().__init__(opt, shared)
        if not SUBWORD_BPE_INSTALLED:
            raise RuntimeError("Please run `pip install subword-nmt`")
        if not opt.get('dict_file'):
            raise RuntimeError('--dict-file is mandatory.')

        self.splitter = re.compile(r'\w+|[^\w\s]', re.UNICODE)

        self.codecs = f"{opt['dict_file']}.codecs"
        if PathManager.exists(self.codecs):
            self._load_from_codecs()

    def add_special_tokens(self, dict_agent, special_tokens: List[str]):
        raise NotImplementedError(
            "--dict-tokenizer BPE does not support special tokens."
        )

    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize the text with bpe if codecs are already finalized.

        Otherwise, returns the regularly split tokens that will train the bpe.

        :param text:
            Raw text to tokenize.
        :return:
            a list of tokens. Will use BPE once finalized.
        """
        text = text.replace('\n', ' __newln__ ')
        tokens = self.splitter.findall(text)

        if hasattr(self, 'bpe'):
            return self.bpe.segment_tokens(tokens)
        else:
            return tokens

    def helper_decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str
    ) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = delimiter.join(tokens)
        text = text.replace('@@ ', '')
        # It's also possible that we get a BPE encoding on the end of the word
        if text.endswith('@@'):
            text = text[:-2]
        text = text.replace('__newln__', '\n')
        return text

    def finalize(
        self, frequencies: Dict[str, int], num_symbols: int = 30000, minfreq: int = 2
    ) -> bool:
        """
        Build the codecs.

        :param frequencies:
            dictionary of (token: frequency) pairs
        :param num_symbols:
            Number of BPE symbols. Recommend 30000-40000.  If <= 0, default
            30000 will be used.
        :param minfreq:
            Minimum frequency of a token before forced BPE decomposition. If <=
            0 will use subword-nmt default of 2.

        :return did_finalize:
            return whether codecs are finalized this call.
        """
        if hasattr(self, 'bpe'):
            # we already finalized the codecs
            return False

        logging.debug(f'Saving bpe codecs to {self.codecs}')

        dictionary = ("{} {}".format(k, v) for k, v in frequencies.items())

        if num_symbols <= 0:
            num_symbols = 30000
        if minfreq <= 0:
            minfreq = 2

        codec_dir, _ = os.path.split(self.codecs)
        PathManager.mkdirs(codec_dir)
        with PathManager.open(self.codecs, 'w', encoding='utf-8') as outstream:
            learn_bpe.learn_bpe(
                dictionary,
                outstream,
                num_symbols=num_symbols,
                min_frequency=minfreq,
                is_dict=True,
            )

        self._load_from_codecs()
        return True

    def _load_from_codecs(self):
        """
        Load BPE from codecs file.
        """
        with PathManager.open(self.codecs, 'r', encoding='utf-8') as codecs_file:
            self.bpe = apply_bpe.BPE(codecs_file)

    def copy_codecs_file(self, target_file: str):
        """
        Copy the codecs file to a new location.

        :param target_file:
            where to copy the codecs.
        """
        with PathManager.open(target_file, 'w', encoding='utf-8') as wfile:
            with PathManager.open(self.codecs, encoding='utf-8') as rfile:
                for line in rfile:
                    wfile.write(line)

    def sync_with_dict(self, dict_agent):
        """
        No need to sync subword BPE.
        """
        pass

    def should_sort(self) -> bool:
        """
        Return whether tokens should be sorted for this particular helper.

        We want to sort with SubwordBPEHelper.
        """
        return True


#######################
# GPT2 BPE            #
# Inspired by Fairseq #
#######################


class Gpt2BpeHelper(BPEHelper):
    """
    BPE Helper for GPT2 Models.

    Original source:
        https://github.com/openai/gpt-2/blob/master/src/encoder.py

    Original license: MIT

    This is a modified implementation from that of fairseq:
        https://github.com/pytorch/fairseq/blob/master/fairseq/data/encoders/gpt2_bpe_utils.py

    Fairseq license: MIT
    """

    DEFAULT_ENCODER_JSON = (
        'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
    )
    DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    ERRORS_METHOD = 'replace'

    def __init__(self, opt: Opt, shared: TShared = None):
        """
        Override init to build the data.
        """
        super().__init__(opt, shared)
        if self.lower:
            warn_once('Are you sure you want to lower case your BPE dictionary?')

        if self.maxtokens > 0 or self.minfreq > 0:
            raise ValueError(
                'You should not filter vocabulary with using --dict-tokenizer bytelevelbpe'
                ' (no --dict-minfreq or --dict-maxtokens).'
            )

        self.bpe_data, self.json_path, self.merge_path = self._build_data()

        # build encoder & decoder
        self.encoder: Dict[str, str] = self._build_encoder(self.json_path)
        self.decoder: Dict[str, str] = {v: k for k, v in self.encoder.items()}

        bpe_merges = [
            tuple(merge_str.split()) for merge_str in self.bpe_data.split('\n')[1:-1]
        ]
        self.byte_encoder = self.bytes_to_unicode()
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

    def _build_data(self) -> Tuple[str, str]:
        """
        Build data.

        Maybe download the appropriate data.

        :return (bpe_data, json_path):
            bpe_data and path to encoder json
        """
        data_path = os.path.join(self.opt['datapath'], 'gpt2')
        vocab_path = os.path.join(data_path, 'vocab.bpe')
        json_path = os.path.join(data_path, 'encoder.json')
        if not PathManager.exists(vocab_path) or not PathManager.exists(json_path):
            make_dir(data_path)
            download(self.DEFAULT_VOCAB_BPE, data_path, 'vocab.bpe')
            download(self.DEFAULT_ENCODER_JSON, data_path, 'encoder.json')
        with PathManager.open(vocab_path, 'r', encoding="utf-8") as f:
            bpe_data = f.read()

        return bpe_data, json_path, vocab_path

    def _build_encoder(self, json_path: str) -> Dict[str, str]:
        """
        Build and return the encoder.

        :param json_path:
            path to encoder json file

        :return:
            encoder, mapping tokens to unicode reps
        """
        with PathManager.open(json_path, 'r', encoding='utf8') as f:
            encoder = json.load(f)
        for each_token in encoder.keys():
            new_token = ''.join(
                # escape nonprintable characters
                '\\' + hex(b).lstrip('0') if (b > 127 or b < 32) else chr(b)
                for b in each_token.encode('utf-8')
            )
            encoder[each_token] = new_token

        return encoder

    @lru_cache()
    def bytes_to_unicode(self) -> Dict[int, str]:
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.

        The reversible bpe codes work on unicode strings. This means you need a large #
        of unicode characters in your vocab if you want to avoid UNKs. When you're at
        something like a 10B token dataset you end up needing around 5K for decent
        coverage. This is a signficant percentage of your normal, say, 32K bpe vocab. To
        avoid that, we want lookup tables between utf-8 bytes and unicode strings. And
        avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs: List[int] = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs: List[int] = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        str_cs: List[str] = [chr(n) for n in cs]
        return dict(zip(bs, str_cs))

    def get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """
        Return set of symbol pairs in a word.

        Word is represented as tuple of symbols (symbols being variable-length strings).

        :param word:
            word to symbolize

        :return pairs:
            set of tuples of symbols
        """
        pairs = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs

    def _dropout_pairs(self, pairs):
        """
        Implements BPE dropout (Provlikov et al., 2019).

        https://arxiv.org/abs/1910.13267

        Randomly removes merges from the list of possible merges. This can
        result in different subwords being used to realized the same string,
        and effectively regularizes representations.
        """
        if not self.bpe_dropout or not self._bpe_dropout_enabled:
            return pairs

        dropped_pairs = [p for p in pairs if random.random() > self.bpe_dropout]
        if not dropped_pairs:
            dropped_pairs = [random.choice(pairs)]
        return dropped_pairs

    def bpe(self, token: str) -> str:
        """
        Convert token to BPE.

        :param token:
            token to convert

        :return bpe_encoding:
            string bpe encoding
        """
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            dropped_pairs = self._dropout_pairs(pairs)
            bigram = min(
                dropped_pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf'))
            )
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
                pairs = self.get_pairs(word)
        return ' '.join(word)

    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        bpe_tokens: List[str] = []
        for token in self.re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')
            )
        return bpe_tokens

    def helper_decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str
    ) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.ERRORS_METHOD
        )
        return text

    def sync_with_dict(self, dict_agent):
        """
        Sync with dictionary agent.

        Just add all of the tokens to the dict

        NOTE: How does this handle special tokens?

        :param dict_agent:
            A DictionaryAgent instantiation
        """
        for each_token in self.encoder.values():
            dict_agent.add_token(each_token)
            dict_agent.freq[each_token] = 1

    def save(self, dir_name: str, file_name: str):
        """
        Save appropriate files.

        :param dir_name:
            directory to save.
        :param file_name:
            file to save.
        """
        out_json_path = os.path.join(dir_name, file_name + "-vocab.json")
        out_merge_path = os.path.join(dir_name, file_name + "-merges.txt")
        # Possibly bad assumption: if the destination file already exists,
        # we don't need to copy it over again.
        if not PathManager.exists(out_json_path):
            logging.info(f"Copying {self.json_path} to {out_json_path}")
            PathManager.copy(self.json_path, out_json_path)
        if not PathManager.exists(out_merge_path):
            logging.info(f"Copying {self.merge_path} to {out_merge_path}")
            PathManager.copy(self.merge_path, out_merge_path)


###################
# HuggingFace BPE #
###################


class HuggingFaceBpeHelper(BPEHelper):
    """
    HuggingFace's ByteLevelBPE Tokenizer.

    Fast because Rust.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__(opt, shared)
        # Default true for HF
        self.special_tok_map = {}  # map from HF
        self.add_prefix_space = opt.get('bpe_add_prefix_space', True)
        if self.add_prefix_space is None:
            self.add_prefix_space = True
        if opt.get('dict_loaded'):
            dfname = opt['dict_file']
            if PathManager.exists(f'{dfname}-merges.txt'):
                opt['bpe_merge'] = f'{dfname}-merges.txt'
            if PathManager.exists(f'{dfname}-vocab.json'):
                opt['bpe_vocab'] = f'{dfname}-vocab.json'
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                'Please install HuggingFace tokenizer with: pip install tokenizers'
            )

        if self.bpe_dropout:
            raise NotImplementedError(
                '--bpe-dropout is not supported with ByteLevelBPE because tokenizers '
                'library does not allow dynamically turning BPE on/off. You can use '
                '--dict-tokenizer slow_bytelevel_bpe to gain this feature.'
            )

        if self.lower:
            warn_once('Are you sure you want to lower case your BPE dictionary?')
        if self.maxtokens > 0 or self.minfreq > 0:
            raise ValueError(
                'You should not filter vocabulary with using --dict-tokenizer bytelevelbpe'
                ' (no --dict-minfreq or --dict-maxtokens).'
            )
        if 'bpe_vocab' not in opt:
            raise ValueError('--bpe-vocab is required for loading pretrained tokenizer')
        if 'bpe_merge' not in opt:
            raise ValueError('--bpe-merge is required for loading pretrained tokenizer')

        self.vocab_path = opt['bpe_vocab']
        self.merge_path = opt['bpe_merge']

        if not self.vocab_path or not self.merge_path:
            raise IOError(
                '--bpe-vocab and --bpe-merge are mandatory with '
                '--dict-tokenizer bytelevelbpe'
            )

        if not PathManager.exists(self.vocab_path):
            raise IOError(
                f'File {self.vocab_path} does not exist. --bpe-vocab must be pretrained.'
            )
        if not PathManager.exists(self.merge_path):
            raise IOError(
                f'File {self.merge_path} does not exist. --bpe-merge must be pretrained.'
            )

        self.tokenizer = ByteLevelBPETokenizer(
            self.vocab_path, self.merge_path, self.add_prefix_space
        )

    def helper_encode(self, text: str) -> List[str]:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        return self.tokenizer.encode(text).tokens

    def helper_decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str
    ) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        return text

    def add_special_tokens(self, dict_agent, special_tokens: List[str]):
        """
        Add special tokens to the tokenizer and dict_agent.
        """
        logging.debug(f'adding the following special tokens: {special_tokens}')
        self.tokenizer.add_special_tokens(special_tokens)  # add to HF

        for tok in special_tokens:
            parlai_key = dict_agent[tok]
            hf_key = self.tokenizer.token_to_id(tok)
            self.special_tok_map[parlai_key] = hf_key

    def sync_with_dict(self, dict_agent):
        """
        Sync the dictionary agent with Hugging Face tokenizer's BPE dict.

        Called only once on initialization.
        """
        special_tokens = [
            dict_agent.null_token,
            dict_agent.start_token,
            dict_agent.end_token,
            dict_agent.unk_token,
        ]
        self.add_special_tokens(dict_agent, special_tokens)

        for i in range(self.tokenizer.get_vocab_size() - len(special_tokens)):
            token = self.tokenizer.id_to_token(i)
            dict_agent.add_token(token)
            # We don't have access to the hugging face word frequency table,
            # just set it to 1 instead
            dict_agent.freq[token] = 1

    def save(self, dir_name: str, file_name: str):
        """
        Save appropriate files.

        :param dir_name:
            directory to save.
        :param file_name:
            file to save.
        """
        self.tokenizer.save_model(dir_name, file_name)


class SlowBytelevelBPE(Gpt2BpeHelper):
    """
    Stand-in for HuggingFace if we do not have access to tokenizers.

    Only EVER used for a model used in interactive mode that was previously trained with
    HF BPE.
    """

    def _build_data(self) -> Tuple[str, str]:
        """
        Override to load dicts if they exist.

        :return (bpe_data, json_path):
            bpe_data and path to encoder json
        """
        bpe_data = None
        json_path = ''
        vocab_path = ''
        if self.opt.get('dict_loaded'):
            dfname = self.opt['dict_file']
            if PathManager.exists(f'{dfname}-merges.txt'):
                vocab_path = f'{dfname}-merges.txt'
            if PathManager.exists(f'{dfname}-vocab.json'):
                json_path = f'{dfname}-vocab.json'

        if PathManager.exists(vocab_path) and PathManager.exists(json_path):
            with PathManager.open(vocab_path, 'r', encoding="utf-8") as f:
                bpe_data = f.read()
        else:
            return super()._build_data()

        return bpe_data, json_path, vocab_path

    def sync_with_dict(self, dict_agent):
        """
        Basically a combination of syncing HF dict with the GPT2 standard.

        It's kinda reversed.

        :param dict_agent:
            Dictionary Agent
        """
        special_tokens = [
            dict_agent.null_token,
            dict_agent.start_token,
            dict_agent.end_token,
            dict_agent.unk_token,
        ]
        dict_agent.tok2ind = {
            tok: i for tok, i in zip(special_tokens, range(len(special_tokens)))
        }
        dict_agent.ind2tok = {v: k for k, v in dict_agent.tok2ind.items()}
        for each_token in self.encoder.values():
            dict_agent.add_token(each_token)
            dict_agent.freq[each_token] = 1
