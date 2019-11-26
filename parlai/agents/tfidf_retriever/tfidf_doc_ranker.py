#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Rank documents with TF-IDF scores.

Adapted from Adam Fisch's work at github.com/facebookresearch/DrQA/
"""

import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import tokenizers
from parlai.utils.logging import logger


class TfidfDocRanker(object):
    """
    Loads a pre-weighted inverted index of token/document terms.

    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        logger.info('Loading %s' % tfidf_path)
        matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata.get('doc_dict', None)
        self.num_docs = self.doc_mat.shape[1] - 1
        self.strict = strict

    def get_doc_index(self, doc_id):
        """
        Convert doc_id --> doc_index.
        """
        return self.doc_dict[0][doc_id] if self.doc_dict else doc_id

    def get_doc_id(self, doc_index):
        """
        Convert doc_index --> doc_id.
        """
        return self.doc_dict[1][doc_index] if self.doc_dict else doc_index

    def closest_docs(self, query, k=1, matrix=None):
        """
        Closest docs by dot product between query and documents in tfidf weighted word
        vector space.

        matrix arg can be provided to be used instead of internal doc matrix.
        """
        spvec = self.text2spvec(query)
        res = spvec * matrix if matrix is not None else spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = res.indices[o_sort]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """
        Process a batch of closest_docs requests multithreaded.

        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """
        Parse the query into tokens (either ngrams or tokens).
        """
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True, filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """
        Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix((data, wids_unique, indptr), shape=(1, self.hash_size))

        return spvec
