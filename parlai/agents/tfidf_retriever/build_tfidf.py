#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A script to build the tf-idf document matrices for retrieval.

Adapted from Adam Fisch's work at github.com/facebookresearch/DrQA/
"""

import numpy as np
import scipy.sparse as sp
import math

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from . import utils
from .doc_db import DocDB
from . import tokenizers
import parlai.utils.logging as logging

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
        logging.info(
            'Data size is too large for scipy to index all of it. '
            'Throwing out {} entries ({}%% of data).'.format(over, pct)
        )
        data = data[:MAX_SZ]
        row = row[:MAX_SZ]
        col = col[:MAX_SZ]
    return data, row, col


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


def get_count_matrix(args, db_opts):
    """
    Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    global MAX_SZ
    with DocDB(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()

    tok_class = tokenizers.get_class(args.tokenizer)
    if args.num_workers is not None and args.num_workers <= 1:
        # single threaded
        init(tok_class, db_opts)
    else:
        workers = ProcessPool(
            args.num_workers, initializer=init, initargs=(tok_class, db_opts)
        )

    # Compute the count matrix in steps (to keep in memory)
    logging.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i : i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logging.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        if args.num_workers is not None and args.num_workers <= 1:
            seq = map(_count, batch)
        else:
            seq = workers.imap_unordered(_count, batch)
        for b_row, b_col, b_data in seq:
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
            if len(data) > MAX_SZ:
                break
        if len(data) > MAX_SZ:
            logging.info('Reached max indexable size, breaking.')
            break

    logging.info('Creating sparse matrix...')
    data, row, col = truncate(data, row, col)

    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids) + 1)
    )
    count_matrix.sum_duplicates()
    return count_matrix


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


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
    logging.info('Counting words...')
    count_matrix = get_count_matrix(args, {'db_path': args.db_path})

    logging.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logging.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    filename = args.out_dir

    logging.info('Saving to %s' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
    }

    utils.save_sparse_csr(filename, tfidf, metadata)
