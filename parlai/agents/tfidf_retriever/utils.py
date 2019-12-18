#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Various retriever utilities.
"""

import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32

try:
    import torch
except ImportError:
    raise ImportError('Need to install Pytorch: go to pytorch.org')


# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def save_sparse_tensor(filename, matrix, metadata=None):
    data = {
        'indices': matrix._indices(),
        'values': matrix._values(),
        'size': matrix.size(),
        'metadata': metadata,
    }
    torch.save(data, filename)


def load_sparse_csr(filename):
    loader = np.load(filename + '.npz', allow_pickle=True)
    matrix = sp.csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']
    )
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


def load_sparse_tensor(filename):
    loader = torch.load(filename)
    matrix = torch.sparse.FloatTensor(
        loader['indices'], loader['values'], loader['size']
    )
    return matrix, loader['metadata'] if 'metadata' in loader else None


# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------


def hash(token, num_buckets):
    """
    Unsigned 32 bit murmurhash for feature hashing.
    """
    return murmurhash3_32(token, positive=True) % num_buckets


# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------


STOPWORDS = {
    'i',
    'me',
    'my',
    'myself',
    'we',
    'our',
    'ours',
    'ourselves',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves',
    'he',
    'him',
    'his',
    'himself',
    'she',
    'her',
    'hers',
    'herself',
    'it',
    'its',
    'itself',
    'they',
    'them',
    'their',
    'theirs',
    'themselves',
    'what',
    'which',
    'who',
    'whom',
    'this',
    'that',
    'these',
    'those',
    'am',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'have',
    'has',
    'had',
    'having',
    'do',
    'does',
    'did',
    'doing',
    'a',
    'an',
    'the',
    'and',
    'but',
    'if',
    'or',
    'because',
    'as',
    'until',
    'while',
    'of',
    'at',
    'by',
    'for',
    'with',
    'about',
    'against',
    'between',
    'into',
    'through',
    'during',
    'before',
    'after',
    'above',
    'below',
    'to',
    'from',
    'up',
    'down',
    'in',
    'out',
    'on',
    'off',
    'over',
    'under',
    'again',
    'further',
    'then',
    'once',
    'here',
    'there',
    'when',
    'where',
    'why',
    'how',
    'all',
    'any',
    'both',
    'each',
    'few',
    'more',
    'most',
    'other',
    'some',
    'such',
    'no',
    'nor',
    'not',
    'only',
    'own',
    'same',
    'so',
    'than',
    'too',
    'very',
    's',
    't',
    'can',
    'will',
    'just',
    'don',
    'should',
    'now',
    'd',
    'll',
    'm',
    'o',
    're',
    've',
    'y',
    'ain',
    'aren',
    'couldn',
    'didn',
    'doesn',
    'hadn',
    'hasn',
    'haven',
    'isn',
    'ma',
    'mightn',
    'mustn',
    'needn',
    'shan',
    'shouldn',
    'wasn',
    'weren',
    'won',
    'wouldn',
    "'ll",
    "'re",
    "'ve",
    "n't",
    "'s",
    "'d",
    "'m",
    "''",
    "``",
}


def normalize(text):
    """
    Resolve different type of unicode encodings.
    """
    if type(text) != str:
        return text
    return unicodedata.normalize('NFD', text)


def filter_word(text):
    """
    Take out english stopwords, punctuation, and compound endings.
    """
    text = normalize(text)
    if regex.match(r'^\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """
    Decide whether to keep or discard an n-gram.

    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)
