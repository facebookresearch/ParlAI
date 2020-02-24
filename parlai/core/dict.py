#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains code for parsing and building a dictionary from text.
"""

from parlai.core.opt import Opt
from parlai.core.build_data import modelzoo_path
from .agents import Agent
from .build_data import make_dir
from collections import defaultdict
from .gpt2_helper import Gpt2BpeHelper
from .hugging_face_bpe_helper import HuggingFaceBpeHelper
import codecs
import copy
import numpy as np
import os
import json
import re

try:
    from subword_nmt import learn_bpe, apply_bpe

    # Don't explicitly throw the runtime error unless the user needs it
    BPE_INSTALLED = True
except ImportError:
    BPE_INSTALLED = False

RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)


def escape(s):
    r"""
    Replace potential special characters with escaped version.

    For example, \n => \\n and \t => \\t

    :param s:
        string to escape
    """
    return s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')


def unescape(s):
    r"""
    Revert escaped characters back to their special version.

    For example, \\n => \n and \\t => \t

    :param s:
        string to unescape
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')


def find_ngrams(token_dict, text, n):
    """
    Break text into ngrams that appear in ``token_dict``.

    :param token_dict:
        ``dict`` to check for ngrams
    :param text:
        ``str`` to look for ngrams in
    :param n:
        ``int`` max size of ngrams
    """
    # base case
    if n <= 1:
        return text
    # tokens committed to output
    saved_tokens = []
    # tokens remaining to be searched in sentence
    search_tokens = text[:]
    # tokens stored until next ngram found
    next_search = []
    while len(search_tokens) >= n:
        ngram = ' '.join(search_tokens[:n])
        if ngram in token_dict:
            # first, search previous unmatched words for smaller ngrams
            sub_n = min(len(next_search), n - 1)
            saved_tokens.extend(find_ngrams(token_dict, next_search, sub_n))
            next_search.clear()
            # then add this ngram
            saved_tokens.append(ngram)
            # then pop this ngram from the remaining words to search
            search_tokens = search_tokens[n:]
        else:
            next_search.append(search_tokens.pop(0))
    remainder = next_search + search_tokens
    sub_n = min(len(remainder), n - 1)
    saved_tokens.extend(find_ngrams(token_dict, remainder, sub_n))
    return saved_tokens


class DictionaryAgent(Agent):
    """
    Builds and/or loads a dictionary.

    The dictionary provides access to the frequency of each token, functions to
    translate sentences from tokens to their vectors (list of ints, each int is the
    index of a token in the dictionary) and back from vectors to tokenized text.
    """

    default_lang = 'english'
    default_maxngram = -1
    default_minfreq = 0
    default_maxtokens = -1
    default_null = '__null__'
    default_start = '__start__'
    default_end = '__end__'
    default_unk = '__unk__'
    default_tok = 're'
    default_lower = False
    default_textfields = 'text,labels'

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add commandline arguments related to the dictionary.
        """
        dictionary = argparser.add_argument_group('Dictionary Arguments')
        dictionary.add_argument(
            '-df',
            '--dict-file',
            help='path to dictionary file. defaults to [model_file].dict if '
            'not set and model_file is set.',
            hidden=True,
        )
        dictionary.add_argument(
            '--dict-initpath',
            hidden=True,
            help='path to a saved dictionary to load tokens / counts from to '
            'seed the dictionary with initial tokens and/or frequencies',
        )
        dictionary.add_argument(
            '--dict-language',
            default=DictionaryAgent.default_lang,
            hidden=True,
            help='sets language for the punkt sentence tokenizer',
        )
        dictionary.add_argument(
            '--dict-max-ngram-size',
            type=int,
            hidden=True,
            default=DictionaryAgent.default_maxngram,
            help='looks for ngrams of up to this size. this is ignored when '
            'building the dictionary. note: this takes approximate '
            'runtime of len(sentence)^max_ngram_size',
        )
        dictionary.add_argument(
            '--dict-minfreq',
            default=DictionaryAgent.default_minfreq,
            type=int,
            help='minimum frequency of words to include them in sorted '
            'dict or minimum frequency of bpe codecs',
            hidden=True,
        )
        dictionary.add_argument(
            '--dict-maxtokens',
            default=DictionaryAgent.default_maxtokens,
            type=int,
            help='max number of tokens to include in dictionary or bpe codecs',
            hidden=True,
        )
        dictionary.add_argument(
            '--dict-nulltoken',
            default=DictionaryAgent.default_null,
            hidden=True,
            help='empty token, can be used for padding or just empty values',
        )
        dictionary.add_argument(
            '--dict-starttoken',
            default=DictionaryAgent.default_start,
            hidden=True,
            help='token for starting sentence generation, if needed',
        )
        dictionary.add_argument(
            '--dict-endtoken',
            default=DictionaryAgent.default_end,
            hidden=True,
            help='token for end of sentence markers, if needed',
        )
        dictionary.add_argument(
            '--dict-unktoken',
            default=DictionaryAgent.default_unk,
            hidden=True,
            help='token to return for unavailable words',
        )
        dictionary.add_argument(
            '-tok',
            '--dict-tokenizer',
            default=DictionaryAgent.default_tok,
            help='Which tokenizer to use. Defaults to "split", which splits '
            'on whitespace as well as recognizing basic punctuation. '
            'Other options include nltk, spacy, gpt2 and bytelevelbpe.',
            hidden=True,
        )
        dictionary.add_argument(
            '--dict-lower',
            default=DictionaryAgent.default_lower,
            type='bool',
            help='Whether or not to lowercase all text seen.',
            hidden=True,
        )
        dictionary.add_argument(
            '--bpe-debug',
            action='store_true',
            hidden=True,
            help='Leave BPE tokens untouched in output. Useful for debugging.',
        )
        dictionary.add_argument(
            '--dict-textfields',
            default=DictionaryAgent.default_textfields,
            hidden=True,
            help='Observation fields which dictionary learns vocabulary from. '
            'Tasks with additional fields may add to this list to handle '
            'any extra vocabulary.',
        )
        dictionary = HuggingFaceBpeHelper.add_cmdline_args(dictionary)
        return dictionary

    def __init__(self, opt: Opt, shared=None):
        """
        Initialize DictionaryAgent.
        """
        self.opt = copy.deepcopy(opt)
        self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)
        self.null_token = opt.get('dict_nulltoken', DictionaryAgent.default_null)
        self.end_token = opt.get('dict_endtoken', DictionaryAgent.default_end)
        self.unk_token = opt.get('dict_unktoken', DictionaryAgent.default_unk)
        self.start_token = opt.get('dict_starttoken', DictionaryAgent.default_start)
        self.max_ngram_size = opt.get(
            'dict_max_ngram_size', DictionaryAgent.default_maxngram
        )
        self.tokenizer = opt.get('dict_tokenizer', DictionaryAgent.default_tok)
        self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        self.maxtokens = opt.get('dict_maxtokens', DictionaryAgent.default_maxtokens)
        self.textfields = opt.get(
            'dict_textfields', DictionaryAgent.default_textfields
        ).split(",")

        try:
            self.tokenizer_fun = getattr(self, self.tokenizer + '_tokenize')
        except AttributeError:
            raise AttributeError(
                'tokenizer type {} not yet supported'.format(self.tokenizer)
            )

        if shared:
            self.freq = shared.get('freq', {})
            self.tok2ind = shared.get('tok2ind', {})
            self.ind2tok = shared.get('ind2tok', {})
        else:
            self.freq = defaultdict(int)
            self.tok2ind = {}
            self.ind2tok = {}

            if self.null_token:
                self.add_token(self.null_token)

            if self.start_token:
                # set special start of sentence word token
                self.add_token(self.start_token)

            if self.end_token:
                # set special end of sentence word token
                self.add_token(self.end_token)

            if self.unk_token:
                # set special unknown word token
                self.add_token(self.unk_token)

            loaded = False
            # If data built via pytorch data teacher, we need to load prebuilt dict
            if opt.get('dict_file'):
                opt['dict_file'] = modelzoo_path(opt.get('datapath'), opt['dict_file'])
                if os.path.isfile(opt['dict_file']):
                    # load pre-existing dictionary
                    self.load(opt['dict_file'])
                    loaded = True

            if not loaded and opt.get('dict_initpath'):
                # load seed dictionary
                opt['dict_initpath'] = modelzoo_path(
                    opt.get('datapath'), opt['dict_initpath']
                )
                # don't check isfile first, should fail if file not found
                self.load(opt['dict_initpath'])

        # cache unk token for later
        self._unk_token_idx = self.tok2ind.get(self.unk_token)

        # initialize tokenizers
        if self.tokenizer == 'nltk':
            try:
                import nltk
            except ImportError:
                raise ImportError('Please install nltk (pip install nltk)')
            # nltk-specific setup
            st_path = 'tokenizers/punkt/{0}.pickle'.format(opt['dict_language'])
            try:
                self.sent_tok = nltk.data.load(st_path)
            except LookupError:
                nltk.download('punkt')
                self.sent_tok = nltk.data.load(st_path)
            self.word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()
        elif self.tokenizer == 'spacy':
            try:
                import spacy
            except ImportError:
                raise ImportError(
                    'Please install spacy and spacy "en" model: '
                    '`pip install -U spacy && '
                    'python -m spacy download en` '
                    'or find alternative installation options '
                    'at spacy.io'
                )
            self.NLP = spacy.load('en')
        elif self.tokenizer == 'bpe':
            if not opt.get('dict_file'):
                raise RuntimeError('--dict-file is mandatory.')
            self.bpehelper = _BPEHelper(f"{opt['dict_file']}.codecs")
        elif self.tokenizer == 'gpt2':
            if self.lower:
                raise ValueError(
                    'Only use --dict-lower false with --dict-tokenizer gpt2'
                )
            if self.maxtokens > 0 or self.minfreq > 0:
                raise ValueError(
                    'You should not filter vocabulary with using --dict-tokenizer gpt2'
                    ' (no --dict-minfreq or --dict-maxtokens).'
                )

            self.gpt2_bpe = Gpt2BpeHelper(opt)
            for each_token in self.gpt2_bpe.list_tokens():
                self.add_token(each_token)
                self.freq[each_token] = 1
        elif self.tokenizer == 'bytelevelbpe':
            if self.lower:
                raise ValueError(
                    'Only use --dict-lower false with --dict-tokenizer bytelevelbpe'
                )
            if self.maxtokens > 0 or self.minfreq > 0:
                raise ValueError(
                    'You should not filter vocabulary with using --dict-tokenizer bytelevelbpe'
                    ' (no --dict-minfreq or --dict-maxtokens).'
                )
            opt_for_byte_level_bpe = copy.copy(opt)
            if loaded:
                dict_basename = os.path.splitext(opt['dict_file'])[0]
                if os.path.isfile('{}-merges.txt'.format(dict_basename)):
                    opt_for_byte_level_bpe['bpe_vocab'] = '{}-vocab.json'.format(
                        dict_basename
                    )
                if os.path.isfile('{}-vocab.json'.format(dict_basename)):
                    opt_for_byte_level_bpe['bpe_merge'] = '{}-merges.txt'.format(
                        dict_basename
                    )
            self.byte_level_bpe = HuggingFaceBpeHelper(opt_for_byte_level_bpe)
            self._sync_bytelevelbpe_dict()

        if not shared:
            if self.null_token:
                # fix count for null token to one billion and three
                self.freq[self.null_token] = 1000000003

            if self.start_token:
                # fix count for start of sentence token to one billion and two
                self.freq[self.start_token] = 1000000002

            if self.end_token:
                # fix count for end of sentence token to one billion and one
                self.freq[self.end_token] = 1000000001

            if self.unk_token:
                # fix count for unknown token to one billion
                self.freq[self.unk_token] = 1000000000

            if opt.get('dict_file'):
                self.save_path = opt['dict_file']

    def add_token(self, word):
        """
        Add a single token to the dictionary.
        """
        if word not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[word] = index
            self.ind2tok[index] = word

    def __contains__(self, key):
        """
        Return if the dictionary contains the key.

        If key is an int, returns whether the key is in the indices. If key is a str,
        return if the token is in the dict of tokens.
        """
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def _word_lookup(self, key):
        # return index from token, or unk_token's index, or None
        return self.tok2ind.get(key, self._unk_token_idx)

    def _index_lookup(self, key):
        # return token from index, or unk_token
        return self.ind2tok.get(key, self.unk_token)

    def __getitem__(self, key):
        """
        Lookup the word or ID.

        If key is an int, returns the corresponding token. If it does not exist, return
        the unknown token. If key is a str, return the token's index. If the token is
        not in the dictionary, return the index of the unknown token. If there is no
        unknown token, return ``None``.
        """
        if type(key) == str:
            return self._word_lookup(key)
        if type(key) == int:
            return self._index_lookup(key)

    def __len__(self):
        return len(self.tok2ind)

    def __setitem__(self, key, value):
        """
        Set the frequency for a word to a value.

        If the key is not in the dictionary, add it to the dictionary and set its
        frequency to value.
        """
        key = str(key)
        if self.lower:
            key = key.lower()
        self.freq[key] = int(value)
        self.add_token(key)

    def keys(self):
        """
        Return all the words in the dictionary.
        """
        return self.tok2ind.keys()

    def copy_dict(self, dictionary):
        """
        Overwrite own state with any state in the other dictionary.

        This allows loading of the contents of another dictionary while keeping the
        current dictionary version.
        """
        for k, v in vars(dictionary).items():
            setattr(self, k, v)

    def max_freq(self):
        """
        Return the largest frequency of any nonspecial token.
        """
        return max(
            self.freq[k]
            for k in self.freq.keys()
            if k
            not in [self.null_token, self.end_token, self.start_token, self.unk_token]
        )

    def freqs(self):
        """
        Return the frequency dictionary.
        """
        # TODO: deprecate this
        return self.freq

    def spacy_tokenize(self, text, **kwargs):
        """
        Tokenize using spaCy.

        Does whatever spaCy does. See https://spacy.io/.
        """
        tokens = self.NLP.tokenizer(text)
        return [t.text for t in tokens]

    def spacy_span_tokenize(self, text):
        """
        Return tuple of tokens, spans.
        """
        # TODO: can we delete this?
        tokens = self.NLP.tokenizer(text)
        return (
            [t.text for t in tokens],
            [(t.idx, t.idx + len(t.text)) for t in tokens],
        )

    def nltk_tokenize(self, text, building=False):
        """
        Tokenize using NLTK PunktTokenizer.

        Uses nltk-trained PunktTokenizer for sentence tokenization and Treebank Word
        Tokenizer for tokenizing words within sentences.
        """
        return (
            token
            for sent in self.sent_tok.tokenize(text)
            for token in self.word_tok.tokenize(sent)
        )

    def gpt2_tokenize(self, text):
        """
        Tokenize using Gpt2 BPE tokenizer.
        """
        return self.gpt2_bpe.encode(text)

    def bytelevelbpe_tokenize(self, text):
        """
        Tokenize using Gpt2 BPE tokenizer.
        """
        return self.byte_level_bpe.encode(text)

    @staticmethod
    def re_tokenize(text):
        r"""
        Tokenize using a liberal regular expression.

        Find boundaries between word characters, newlines, and non-word
        non-whitespace tokens ``(r'[\\w\\n]+ | [^\\w\\s] | \\n')``.

        This splits along whitespace and punctuation and keeps the newline as
        a token in the returned list.
        """
        return RETOK.findall(text)

    @staticmethod
    def split_tokenize(text):
        """
        Tokenize on whitespace and some limited punctuation.

        Splits tokens based on whitespace after adding whitespace around
        punctuation.

        Use re_tokenize if you want more robust handling of punctuation.
        """
        return (
            text.replace('.', ' . ')
            .replace(',', ' , ')
            .replace(';', ' ; ')
            .replace(':', ' : ')
            .replace('!', ' ! ')
            .replace('?', ' ? ')
            .split()
        )

    @staticmethod
    def space_tokenize(text):
        """
        Tokenize exactly on spaces.

        Useful when text is pre-tokenized.
        """
        return text.strip().split(' ')

    def span_tokenize(self, text):
        """
        Tokenize and find  starting index of each token in the original string.
        """
        # TODO: can this be deleted?
        if self.tokenizer == 'spacy':
            # spacy has own
            return self.spacy_span_tokenize(text)
        tokens = self.tokenize(text)
        curr_idx = 0
        indices = []
        for t in tokens:
            while text[curr_idx] != t[0]:
                curr_idx += 1
            indices.append((curr_idx, curr_idx + len(t)))
            curr_idx += len(t)
        return tokens, indices

    def tokenize(self, text, building=False):
        """
        Return a sequence of tokens from the iterable.
        """
        if self.lower:
            text = text.lower()

        # calls the selected tokenizer function e.g. 're' => re_tokenize(text)
        word_tokens = self.tokenizer_fun(text)

        if not building and self.max_ngram_size > 1:
            # search for ngrams during parse-time
            # TODO(ahm): support build-time ngrams using word2vec heuristic?
            word_tokens = find_ngrams(self.tok2ind, word_tokens, self.max_ngram_size)
        return word_tokens

    def bpe_tokenize(self, text):
        """
        Return a sequence of BPE-tokens from the text.
        """
        return self.bpehelper.tokenize(text)

    def add_to_dict(self, tokens):
        """
        Build dictionary from the list of provided tokens.
        """
        self.built = False
        for token in tokens:
            self.add_token(token)
            self.freq[token] += 1

    def remove_tail(self, min_freq):
        """
        Remove elements below the frequency cutoff from the dictionary.
        """
        to_remove = []
        for token, freq in self.freq.items():
            if freq < min_freq:
                # queue up removals since can't mutate dict during iteration
                to_remove.append(token)

        for token in to_remove:
            del self.freq[token]
            idx = self.tok2ind.pop(token)
            del self.ind2tok[idx]

    def _remove_non_bpe(self):
        """
        Set the dictionary vocab to the bpe vocab, merging counts.
        """
        to_remove = []
        to_add = []
        for token, freq in self.freq.items():
            tokens = self.bpe_tokenize(token)
            if len(tokens) != 1:
                for t in tokens:
                    to_add.append((t, freq))
                to_remove.append(token)
        for token in to_remove:
            del self.freq[token]
            idx = self.tok2ind.pop(token)
            del self.ind2tok[idx]
        for token, freq in to_add:
            self.add_token(token)
            self.freq[token] += freq

    def resize_to_max(self, maxtokens):
        """
        Trims the dictionary to the maximum number of tokens.
        """
        if maxtokens >= 0 and len(self.tok2ind) > maxtokens:
            for k in range(maxtokens, len(self.ind2tok)):
                v = self.ind2tok[k]
                del self.ind2tok[k]
                del self.tok2ind[v]
                del self.freq[v]

    def load(self, filename):
        """
        Load pre-existing dictionary in 'token[<TAB>count]' format.

        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        print('Dictionary: loading dictionary from {}'.format(filename))

        lower_special = self.null_token == self.null_token.lower()
        SPECIAL_TOKENS = {'__UNK__', '__NULL__', '__END__', '__START__'}
        with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as read:
            for line in read:
                split = line.strip().split('\t')
                token = unescape(split[0])
                if lower_special and token in SPECIAL_TOKENS:
                    token = token.lower()
                cnt = int(split[1]) if len(split) > 1 else 0
                self.freq[token] = cnt
                self.add_token(token)
        print('[ num words =  %d ]' % len(self))

    def save(self, filename=None, append=False, sort=True):
        """
        Save dictionary to file.

        Format is 'token<TAB>count' for every token in the dictionary, sorted
        by count with the most frequent words first.

        If ``append`` (default ``False``) is set to ``True``, appends instead of
        overwriting.

        If ``sort`` (default ``True``), then first sort the dictionary before saving.
        """
        filename = self.opt['dict_file'] if filename is None else filename

        if self.tokenizer == 'bpe':
            needs_removal = self.bpehelper.finalize(
                self.freq, num_symbols=self.maxtokens, minfreq=self.minfreq
            )
            if needs_removal:
                self._remove_non_bpe()
            elif filename != self.opt['dict_file']:
                # need to copy over the old codecs file
                self.bpehelper.copy_codecs_file(filename + '.codecs')
            if sort:
                self.sort(trim=False)
        elif self.tokenizer == 'gpt2':
            # never remove or sort tokens from gpt2
            pass
        elif self.tokenizer == 'bytelevelbpe':
            # never remove or sort tokens from bytelevelbpe, it should be the same as the hugging face tokenizer
            pass
        elif sort:
            self.sort(trim=True)

        print('Dictionary: saving dictionary to {}'.format(filename))

        make_dir(os.path.dirname(filename))
        mode = 'a' if append else 'w'
        with open(filename, mode, encoding='utf-8') as write:
            for i in self.ind2tok.keys():
                tok = self.ind2tok[i]
                cnt = self.freq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))

        # save opt file
        with open(filename + '.opt', 'w', encoding='utf-8') as handle:
            json.dump(self.opt, handle, indent=4)
        # save the byte level bpe model file as well
        if self.tokenizer == 'bytelevelbpe':
            # This saves filename-vocab.json and finlename-merges.txt as hugging face tokenizer do
            self.byte_level_bpe.tokenizer.save(
                os.path.dirname(filename), os.path.splitext(filename)[0]
            )

    def sort(self, trim=True):
        """
        Sort the dictionary.

        Inline operation. Rearranges the dictionary so that the elements with
        the lowest index have the highest counts. This reindexes the dictionary
        according to the sorted frequencies, breaking ties alphabetically by
        token.

        :param bool trim:
            If True, truncate the dictionary based on minfreq and maxtokens.
        """
        if trim and self.tokenizer == 'gpt2':
            raise RuntimeError("You should not trim the dictionary when using gpt-2.")
        if trim and self.tokenizer == 'bytelevelbpe':
            raise RuntimeError(
                "You should not trim the dictionary when using bytelevelbpe."
            )
        # sort first by count, then alphabetically
        if trim:
            self.remove_tail(self.minfreq)
        sorted_pairs = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        new_tok2ind = {}
        new_ind2tok = {}
        for i, (tok, _) in enumerate(sorted_pairs):
            new_tok2ind[tok] = i
            new_ind2tok[i] = tok
        self.tok2ind = new_tok2ind
        self.ind2tok = new_ind2tok
        if trim:
            self.resize_to_max(self.maxtokens)
        assert len(self.freq) == len(self.ind2tok) == len(self.tok2ind)
        return sorted_pairs

    def parse(self, txt_or_vec, vec_type=list):
        """
        Parse either text or a vector of indices.

        Calls `~txt2vec` if `txt_or_vec is a string, or `~vec2txt` otherwise.

        :param vec_type:
            type of the returned vector if the input is a string.
        """
        # TODO: try to deprecate this, preferring straight txt2vec
        if type(txt_or_vec) == str:
            return self.txt2vec(txt_or_vec, vec_type)
        else:
            return self.vec2txt(txt_or_vec)

    def txt2vec(self, text, vec_type=list):
        """
        Convert a string to a vector (list of ints).

        First runs a sentence tokenizer, then a word tokenizer.

        :param type vec_type:
            The type of the returned vector if the input is a string. Suggested
            ``list``, ``tuple``, ``set``, or ``np.ndarray``.
        """
        itr = (self._word_lookup(token) for token in self.tokenize(str(text)))
        if vec_type == list or vec_type == tuple or vec_type == set:
            res = vec_type(itr)
        elif vec_type == np.ndarray:
            res = np.fromiter(itr, np.int)
        else:
            raise RuntimeError('Type {} not supported by dict'.format(vec_type))
        return res

    def vec2txt(self, vector, delimiter=' '):
        """
        Convert a vector of IDs to a string.

        Converts a vector (iterable of ints) into a string, with each token separated by
        the delimiter (default ``' '``).
        """
        if self.tokenizer == 'gpt2' and not self.opt.get('bpe_debug', False):
            return self.gpt2_bpe.decode(self[int(idx)] for idx in vector)
        if self.tokenizer == 'bytelevelbpe' and not self.opt.get('bpe_debug', False):
            # We add special tokens in the beginning of ParlAI dict but in the
            # end of Hugging Face dict,there is an offset of 4 between them.
            vector = [
                idx + len(self.tok2ind) - 4 if idx < 4 else idx - 4 for idx in vector
            ]
            text = self.byte_level_bpe.tokenizer.decode(vector)
            if self.byte_level_bpe.add_prefix_space:
                assert text.startswith(' ')
                text = text.lstrip(' ')
            return text
        # if we want to debug into this gpt2 bpe, you will get next line
        text = delimiter.join(self[int(idx)] for idx in vector)
        # if we used a BPE tokenizer we need to rejoin the encodings
        if self.tokenizer == 'bpe' and not self.opt.get('bpe_debug', False):
            text = text.replace('@@ ', '')
            # It's also possible that we get a BPE encoding on the end of the word
            if text.endswith('@@'):
                text = text[:-2]
            text = text.replace('__newln__', '\n')
        return text

    def act(self):
        """
        Add words in the last observation to the dictionary.

        This checks any fields in the message present in the --dict-textfields argument
        (e.g. "text,labels").
        """
        for textfield in self.textfields:
            source = self.observation.get(textfield)
            if source is None:
                continue
            # fields may be singleton strings or lists of strings.
            # wrap the singleton strings in a list to iterate over them
            if type(source) is str:
                source = [source]
            for text in source:
                if text:
                    self.add_to_dict(self.tokenize(text))
        return {'id': 'Dictionary'}

    def share(self):
        """
        Share internal dicts.
        """
        shared = super().share()
        shared['freq'] = self.freq
        shared['tok2ind'] = self.tok2ind
        shared['ind2tok'] = self.ind2tok
        return shared

    def shutdown(self):
        """
        Save on shutdown if ``save_path`` is set.
        """
        if hasattr(self, 'save_path'):
            self.save(self.save_path)

    def __str__(self):
        """
        Return string representation of frequencies in dictionary.
        """
        return str(self.freq)

    def _sync_bytelevelbpe_dict(self):
        """
        Sync the dictionary agent with Hugging Face tokenizer's BPE dict.

        Called only once on initialization.
        """
        if self.tokenizer != 'bytelevelbpe':
            return
        special_tokens = [
            self.null_token,
            self.start_token,
            self.end_token,
            self.unk_token,
        ]
        self.byte_level_bpe.tokenizer.add_special_tokens(special_tokens)
        for i in range(self.byte_level_bpe.tokenizer.get_vocab_size() - 4):
            token = self.byte_level_bpe.tokenizer.id_to_token(i)
            self.add_token(token)
            # We don't have access to the hugging face word frequency table, just set it to 1 instead
            self.freq[token] = 1


class _BPEHelper(object):
    """
    Helper class for performing BPE subword tokenization.

    For technical details, please refer to https://arxiv.org/abs/1508.07909.
    This class just wraps around the official subword-nmt repository.

    This API expects the user to call tokenize() onto the training data,
    then call finalize() to learn the encodings, and then iterate over the data
    in a second pass, calling tokenize() again to get processed output.
    """

    def __init__(self, codecs_filename):
        """
        Initialize the BPE module.

        If `codecs_filename` already exists, loads the pretrained codecs.
        If it does not, codecs will be saved there after a call to `finalize()`.

        :param codecs_filename:
            place to save/load codecs.
        """
        if not BPE_INSTALLED:
            raise RuntimeError(
                "Please run \"pip install 'git+https://github.com/rsennrich"
                "/subword-nmt.git#egg=subword-nmt'\""
            )

        self.splitter = re.compile(r'\w+|[^\w\s]', re.UNICODE)

        self.codecs = codecs_filename
        if os.path.exists(self.codecs):
            self._load_from_codecs()

    def _load_from_codecs(self):
        with open(self.codecs, 'r', encoding='utf-8') as codecs_file:
            self.bpe = apply_bpe.BPE(codecs_file)

    def tokenize(self, text):
        """
        Tokenize the text with bpe if codecs are already finalized.

        Otherwise, returns the regularly split tokens that will train the bpe.

        :param text: str. Raw text to tokenize.
        :return: a list of tokens. Will use BPE once finalized.
        """
        text = text.replace('\n', ' __newln__ ')
        tokens = self.splitter.findall(text)

        if hasattr(self, 'bpe'):
            return self.bpe.segment_tokens(tokens)
        else:
            return tokens

    def finalize(self, frequencies, num_symbols=30000, minfreq=2):
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
        """
        if hasattr(self, 'bpe'):
            # we already finalized the codecs
            return False

        print('Dictionary: saving bpe codecs to {}'.format(self.codecs))

        dictionary = ("{} {}".format(k, v) for k, v in frequencies.items())

        if num_symbols <= 0:
            num_symbols = 30000
        if minfreq <= 0:
            minfreq = 2

        codec_dir, _ = os.path.split(self.codecs)
        os.makedirs(codec_dir, exist_ok=True)
        with open(self.codecs, 'w', encoding='utf-8') as outstream:
            learn_bpe.learn_bpe(
                dictionary,
                outstream,
                num_symbols=num_symbols,
                min_frequency=minfreq,
                is_dict=True,
            )

        self._load_from_codecs()
        return True

    def copy_codecs_file(self, target_file):
        """
        Copy the codecs file to a new location.
        """
        with open(target_file, 'w', encoding='utf-8') as wfile:
            with open(self.codecs, encoding='utf-8') as rfile:
                for line in rfile:
                    wfile.write(line)
