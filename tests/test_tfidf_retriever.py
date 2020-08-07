#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


class TestTfidfRetriever(unittest.TestCase):
    """
    Basic tests on the display_data.py example.
    """

    def test_sparse_tfidf_retriever(self):
        testing_utils.train_model(
            dict(
                model='tfidf_retriever',
                task='babi:task1k:1',
                retriever_numworkers=4,
                retriever_hashsize=2 ** 8,
                datatype='train:ordered',
                num_epochs=1,
            )
        )

    def test_ptb_tok(self):
        """
        Tests with the PTB tokenizer.
        """
        testing_utils.train_model(
            dict(
                model='tfidf_retriever',
                task='babi:task1k:1',
                retriever_numworkers=4,
                retriever_hashsize=2 ** 8,
                datatype='train:ordered',
                num_epochs=1,
                retriever_tokenizer='regexp',
            )
        )

    def test_simple_tok(self):
        """
        Tests with the simple tokenizer.
        """
        testing_utils.train_model(
            dict(
                model='tfidf_retriever',
                task='babi:task1k:1',
                retriever_numworkers=4,
                retriever_hashsize=2 ** 8,
                datatype='train:ordered',
                num_epochs=1,
                retriever_tokenizer='simple',
            )
        )


if __name__ == '__main__':
    unittest.main()
