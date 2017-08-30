# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Contains code for parsing and building a dictionary from text."""

from .agents import Agent
from collections import defaultdict
import copy
import numpy as np
import nltk
import os
import re


def escape(s):
    """Replace potential special characters with escaped version.
    For example, newline => \\n and tab => \\t
    """
    return s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')


def unescape(s):
    """Revert escaped characters back to their special version.
    For example, \\n => newline and \\t => tab
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')


def find_ngrams(token_dict, text, n):
    """Breaks text into ngrams that appear in ``token_dict``."""
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
    """Builds and/or loads a dictionary.

    The dictionary provides access to the frequency of each token, functions
    to translate sentences from tokens to their vectors (list of ints, each
    int is the index of a token in the dictionary) and back from vectors to
    tokenized text.
    """

    default_lang = 'english'
    default_maxngram = -1
    default_minfreq = 0
    default_null = '__NULL__'
    default_end = '__END__'
    default_unk = '__UNK__'
    default_start = '__START__'

    @staticmethod
    def add_cmdline_args(argparser):
        dictionary = argparser.add_argument_group('Dictionary Arguments')
        dictionary.add_argument(
            '--dict-file',
            help='if set, the dictionary will automatically save to this path' +
                 ' during shutdown')
        dictionary.add_argument(
            '--dict-initpath',
            help='path to a saved dictionary to load tokens / counts from to ' +
                 'seed the dictionary with initial tokens and/or frequencies')
        dictionary.add_argument(
            '--dict-language', default=DictionaryAgent.default_lang,
            help='sets language for the punkt sentence tokenizer')
        dictionary.add_argument(
            '--dict-max-ngram-size', type=int,
            default=DictionaryAgent.default_maxngram,
            help='looks for ngrams of up to this size. this is ignored when ' +
                 'building the dictionary. note: this takes approximate ' +
                 'runtime of len(sentence)^max_ngram_size')
        dictionary.add_argument(
            '--dict-minfreq', default=DictionaryAgent.default_minfreq, type=int,
            help='minimum frequency of words to include them in the dictionary')
        dictionary.add_argument(
           '--dict-nulltoken', default=DictionaryAgent.default_null,
           help='empty token, can be used for padding or just empty values')
        dictionary.add_argument(
           '--dict-endtoken', default=DictionaryAgent.default_end,
           help='token for end of sentence markers, if needed')
        dictionary.add_argument(
            '--dict-unktoken', default=DictionaryAgent.default_unk,
            help='token to return for unavailable words')
        dictionary.add_argument(
           '--dict-starttoken', default=DictionaryAgent.default_start,
           help='token for starting sentence generation, if needed')
        dictionary.add_argument(
            '--dict-maxexs', default=100000, type=int,
            help='max number of examples to build dict on')
        return dictionary

    def __init__(self, opt, shared=None):
        # initialize fields
        self.opt = copy.deepcopy(opt)
        self.null_token = opt['dict_nulltoken']
        self.end_token = opt['dict_endtoken']
        self.unk_token = opt['dict_unktoken']
        self.start_token = opt['dict_starttoken']
        self.max_ngram_size = opt['dict_max_ngram_size']

        if shared:
            self.freq = shared.get('freq', {})
            self.tok2ind = shared.get('tok2ind', {})
            self.ind2tok = shared.get('ind2tok', {})
        else:
            self.freq = defaultdict(int)
            self.tok2ind = {}
            self.ind2tok = {}

            if self.null_token:
                self.tok2ind[self.null_token] = 0
                self.ind2tok[0] = self.null_token

            if self.end_token:
                # set special end of sentence word token
                index = len(self.tok2ind)
                self.tok2ind[self.end_token] = index
                self.ind2tok[index] = self.end_token

            if self.unk_token:
                # set special unknown word token
                index = len(self.tok2ind)
                self.tok2ind[self.unk_token] = index
                self.ind2tok[index] = self.unk_token

            if self.start_token:
                # set special start of sentence word token
                index = len(self.tok2ind)
                self.tok2ind[self.start_token] = index
                self.ind2tok[index] = self.start_token

            if opt.get('dict_file') and os.path.isfile(opt['dict_file']):
                # load pre-existing dictionary
                self.load(opt['dict_file'])
            elif opt.get('dict_initpath'):
                # load seed dictionary
                self.load(opt['dict_initpath'])


        # initialize tokenizers
        st_path = 'tokenizers/punkt/{0}.pickle'.format(opt['dict_language'])
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download('punkt')
            self.sent_tok = nltk.data.load(st_path)

        self.word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()

        if not shared:

            if self.start_token:
                # fix count for start of sentence token to one billion and three
                self.freq[self.start_token] = 1000000003

            if self.null_token:
                # fix count for null token to one billion and two
                self.freq[self.null_token] = 1000000002

            if self.end_token:
                # fix count for end of sentence token to one billion and one
                self.freq[self.end_token] = 1000000001

            if self.unk_token:
                # fix count for unknown token to one billion
                self.freq[self.unk_token] = 1000000000

            if opt.get('dict_file'):
                self.save_path = opt['dict_file']

    def __contains__(self, key):
        """If key is an int, returns whether the key is in the indices.
        If key is a str, return if the token is in the dict of tokens.
        """
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def __getitem__(self, key):
        """If key is an int, returns the corresponding token. If it does not
        exist, return the unknown token.
        If key is a str, return the token's index. If the token is not in the
        dictionary, return the index of the unknown token. If there is no
        unknown token, return ``None``.
        """
        if type(key) == int:
            # return token from index, or unk_token
            return self.ind2tok.get(key, self.unk_token)
        elif type(key) == str:
            # return index from token, or unk_token's index, or None
            return self.tok2ind.get(key, self.tok2ind.get(self.unk_token, None))

    def __len__(self):
        return len(self.tok2ind)

    def __setitem__(self, key, value):
        """If the key is not in the dictionary, add it to the dictionary and set
        its frequency to value.
        """
        key = str(key)
        self.freq[key] = int(value)
        if key not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[key] = index
            self.ind2tok[index] = key

    def freqs(self):
        return self.freq

    def _sent_tokenize(self, text, building=False):
        """Uses nltk-trained PunktTokenizer for sentence tokenization"""
        text = text.replace('|', ' ' if building else ' __pipe__ ')
        return self.sent_tok.tokenize(text)

    def _word_tokenize(self, text, building=False):
        """Uses nltk Treebank Word Tokenizer for tokenizing words within
        sentences.
        """
        word_tokens = self.word_tok.tokenize(text)

        if not building and self.max_ngram_size > 1:
            # search for ngrams during parse-time
            # TODO(ahm): support build-time ngrams using word2vec heuristic?
            word_tokens = find_ngrams(self.tok2ind, word_tokens,
                                      self.max_ngram_size)
        return word_tokens

    def tokenize(self, text, building=False):
        """Returns a sequence of tokens from the iterable."""
        return (token for sent in self._sent_tokenize(text, building)
                for token in self._word_tokenize(sent, building))

    def add_to_dict(self, tokens):
        """ Builds dictionary from the list of provided tokens."""
        for token in tokens:
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def remove_tail(self, min_freq):
        to_remove = []
        for token, freq in self.freq.items():
            if freq < min_freq:
                # queue up removals since can't mutate dict during iteration
                to_remove.append(token)
                # other dicts can be modified as we go
                idx = self.tok2idx.pop(token)
                del self.ind2tok[idx]
        for token in to_remove:
            del self.freq[token]

    def load(self, filename):
        """Load pre-existing dictionary in 'token[<TAB>count]' format.
        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        print('Dictionary: loading dictionary from {}'.format(
              filename))
        with open(filename) as read:
            for line in read:
                split = line.strip().split('\t')
                token = unescape(split[0])
                cnt = int(split[1]) if len(split) > 1 else 0
                self.freq[token] = cnt
                if token not in self.tok2ind:
                    index = len(self.tok2ind)
                    self.tok2ind[token] = index
                    self.ind2tok[index] = token
        print('[ num words =  %d ]' % len(self))

    def save(self, filename=None, append=False, sort=True):
        """Save dictionary to file.
        Format is 'token<TAB>count' for every token in the dictionary, sorted
        by count with the most frequent words first.

        If ``append`` (default ``False``) is set to ``True``, appends instead of
        overwriting.

        If ``sort`` (default ``True``), then first sort the dictionary before saving.
        """
        filename = self.opt['dict_file'] if filename is None else filename
        print('Dictionary: saving dictionary to {}'.format(filename))
        if sort:
            self.sort()

        with open(filename, 'a' if append else 'w') as write:
            for i in range(len(self.ind2tok)):
                tok = self.ind2tok[i]
                cnt = self.freq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))

    def sort(self):
        """Sorts the dictionary, so that the elements with the lowest index have
        the highest counts. This reindexes the dictionary according to the
        sorted frequencies, breaking ties alphabetically by token.
        """
        # sort first by count, then alphabetically
        sorted_pairs = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        new_tok2ind = {}
        new_ind2tok = {}
        for i, (tok, _) in enumerate(sorted_pairs):
            new_tok2ind[tok] = i
            new_ind2tok[i] = tok
        self.tok2ind = new_tok2ind
        self.ind2tok = new_ind2tok
        return sorted_pairs

    def parse(self, txt_or_vec, vec_type=list):
        """Convenience function for parsing either text or vectors of indices.

        ``vec_type`` is the type of the returned vector if the input is a string.
        """
        if type(txt_or_vec) == str:
            res = self.txt2vec(txt_or_vec, vec_type)
            assert type(res) == vec_type
            return res
        else:
            return self.vec2txt(txt_or_vec)

    def txt2vec(self, text, vec_type=list):
        """Converts a string to a vector (list of ints).

        First runs a sentence tokenizer, then a word tokenizer.

        ``vec_type`` is the type of the returned vector if the input is a string.
        """
        if vec_type == np.ndarray:
            res = np.fromiter(
                (self[token] for token in self.tokenize(str(text))),
                np.int
            )
        elif vec_type == list or vec_type == tuple or vec_type == set:
            res = vec_type((self[token] for token in self.tokenize(str(text))))
        else:
            raise RuntimeError('Type {} not supported by dict'.format(vec_type))
        assert type(res) == vec_type
        return res

    def vec2txt(self, vector, delimiter=' '):
        """Converts a vector (iterable of ints) into a string, with each token
        separated by the delimiter (default ``' '``).
        """
        return delimiter.join(self[int(idx)] for idx in vector)

    def act(self):
        """Add any words passed in the 'text' field of the observation to this
        dictionary.
        """
        for source in ([self.observation.get('text')],
                        self.observation.get('labels')):
            if source:
                for text in source:
                    if text:
                        self.add_to_dict(self.tokenize(text))
        return {'id': 'Dictionary'}

    def share(self):
        shared = {}
        shared['freq'] = self.freq
        shared['tok2ind'] = self.tok2ind
        shared['ind2tok'] = self.ind2tok
        shared['opt'] = self.opt
        shared['class'] = type(self)
        return shared

    def shutdown(self):
        """Save on shutdown if ``save_path`` is set."""
        if hasattr(self, 'save_path'):
            self.save(self.save_path)

    def __str__(self):
        return str(self.freq)
