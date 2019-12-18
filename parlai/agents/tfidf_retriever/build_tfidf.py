#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A script to build the tf-idf document matrices for retrieval.

Adapted from Adam Fisch's work at github.com/facebookresearch/DrQA/
"""

import torch
import numpy as np
import scipy.sparse as sp
import argparse
import os
import math

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from . import utils
from .doc_db import DocDB
from . import tokenizers
from parlai.utils.logging import logger

fmt = '%(asctime)s: [ %(message)s ]'
logger.set_format(fmt)
# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


MAX_SZ = int(math.pow(2, 30) * 1.8)

# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def truncate(data, row, col):
    global MAX_SZ
    if len(data) > MAX_SZ:
        over = len(data) - MAX_SZ
        pct = over / len(data)
        logger.info(
            'Data size is too large for scipy to index all of it. '
            'Throwing out {} entries ({}%% of data).'.format(over, pct)
        )
        data = data[:MAX_SZ]
        row = row[:MAX_SZ]
        col = col[:MAX_SZ]
    return data, row, col


def sparse_nonzero(sparse_t):
    return sparse_t.coalesce()._indices()


def sparse_sum(sparse_t):
    return sparse_t._values().sum()


def sparse_log1p(sparse_t):
    t = sparse_t.coalesce()
    t._values().log1p_()
    return t


def live_count_matrix(args, cands):
    global PROCESS_TOK
    if PROCESS_TOK is None:
        PROCESS_TOK = tokenizers.get_class(args.tokenizer)()
    row, col, data = [], [], []
    for i, c in enumerate(cands):
        cur_row, cur_col, cur_data = count_text(args.ngram, args.hash_size, i, c)
        row += cur_row
        col += cur_col
        data += cur_data

    data, row, col = truncate(data, row, col)
    count_matrix = sp.csr_matrix((data, (row, col)), shape=(args.hash_size, len(cands)))
    count_matrix.sum_duplicates()
    return count_matrix


def live_count_matrix_t(args, cands):
    global PROCESS_TOK
    if PROCESS_TOK is None:
        PROCESS_TOK = tokenizers.get_class(args.tokenizer)()
    row, col, data = [], [], []
    for i, c in enumerate(cands):
        cur_row, cur_col, cur_data = count_text(args.ngram, args.hash_size, i, c)
        row += cur_row
        col += cur_col
        data += cur_data

    count_matrix = torch.sparse.FloatTensor(
        torch.LongTensor([row, col]),
        torch.FloatTensor(data),
        torch.Size([args.hash_size, len(cands)]),
    ).coalesce()
    return count_matrix


def count_text(ngram, hash_size, doc_id, text=None):
    """
    Compute hashed ngram counts of text.
    """
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(utils.normalize(text))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(n=ngram, uncased=True, filter_fn=utils.filter_ngram)

    # Hash ngrams and count occurences
    counts = Counter([utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([doc_id] * len(counts))
    data.extend(counts.values())
    return row, col, data


def count(ngram, hash_size, doc_id):
    """
    Fetch the text of a document and compute hashed ngrams counts.
    """
    return count_text(ngram, hash_size, doc_id, text=fetch_text(doc_id))


def get_count_matrix_t(args, db_opts):
    """
    Form a sparse word to document count matrix (inverted index, torch ver).

    M[i, j] = # times word i appears in document j.
    """
    global MAX_SZ
    with DocDB(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers, initializer=init, initargs=(tok_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i : i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = torch.sparse.FloatTensor(
        torch.LongTensor([row, col]),
        torch.FloatTensor(data),
        torch.Size([args.hash_size, len(doc_ids) + 1]),
    ).coalesce()
    return count_matrix


def get_count_matrix(args, db_opts):
    """
    Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    global MAX_SZ
    with DocDB(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers, initializer=init, initargs=(tok_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i : i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
            if len(data) > MAX_SZ:
                break
        if len(data) > MAX_SZ:
            logger.info('Reached max indexable size, breaking.')
            break
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    data, row, col = truncate(data, row, col)

    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids) + 1)
    )
    count_matrix.sum_duplicates()
    return count_matrix


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix_t(cnts):
    """
    Convert the word count matrix into tfidf one (torch version).

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Nt = get_doc_freqs_t(cnts)
    idft = ((cnts.size(1) - Nt + 0.5) / (Nt + 0.5)).log()
    idft[idft < 0] = 0
    tft = sparse_log1p(cnts)
    inds, vals = tft._indices(), tft._values()
    for i, ind in enumerate(inds[0]):
        vals[i] *= idft[ind]
    tfidft = tft
    return tfidft


def get_tfidf_matrix(cnts):
    """
    Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Nt = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Nt + 0.5) / (Nt + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)

    return tfidfs


def get_doc_freqs_t(cnts):
    """
    Return word --> # of docs it appears in (torch version).
    """
    return torch.histc(
        cnts._indices()[0].float(), bins=cnts.size(0), min=0, max=cnts.size(0)
    )


def get_doc_freqs(cnts):
    """
    Return word --> # of docs it appears in.
    """
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def run(args):
    # ParlAI version of run method, modified slightly
    logger.info('Counting words...')
    count_matrix = get_count_matrix(args, {'db_path': args.db_path})

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    filename = args.out_dir

    logger.info('Saving to %s' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
    }

    utils.save_sparse_csr(filename, tfidf, metadata)


if __name__ == '__main__':
    # not used in ParlAI but kept for reference
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'db_path',
        type=str,
        default=None,
        help='Path to sqlite db holding document texts',
    )
    parser.add_argument(
        'out_dir', type=str, default=None, help='Directory for saving output files'
    )
    parser.add_argument(
        '--ngram',
        type=int,
        default=2,
        help=('Use up to N-size n-grams ' '(e.g. 2 = unigrams + bigrams)'),
    )
    parser.add_argument(
        '--hash-size',
        type=int,
        default=int(math.pow(2, 24)),
        help='Number of buckets to use for hashing ngrams',
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='simple',
        help=("String option specifying tokenizer type to use " "(e.g. 'corenlp')"),
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of CPU processes (for tokenizing, etc)',
    )
    args = parser.parse_args()

    logger.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix(args, {'db_path': args.db_path})

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += '-tfidf-ngram=%d-hash=%d-tokenizer=%s' % (
        args.ngram,
        args.hash_size,
        args.tokenizer,
    )
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict,
    }
    utils.save_sparse_csr(filename, tfidf, metadata)
