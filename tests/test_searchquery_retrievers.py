#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.params import ParlaiParser
from parlai.agents.rag.dpr import BertTokenizerDictionaryAgent
from parlai.agents.rag.retrievers import (
    SearchQuerySearchEngineRetriever,
    SearchQueryFAISSIndexRetriever,
    Document,
    NO_SEARCH_QUERY,
)
from parlai.agents.rag.retrieve_api import (
    SearchEngineRetrieverMock,
    SearchEngineRetriever,
)
import numpy as np
import torch
import unittest


################################################################
# Search Engine FiD Agent
################################################################


class MockSearchQuerySearchEngineRetriever(SearchQuerySearchEngineRetriever):
    def init_search_query_generator(self, opt):
        pass

    def generate_search_query(self, query):
        return ['mock search query', NO_SEARCH_QUERY]

    def initiate_retriever_api(self, opt) -> SearchEngineRetriever:
        return SearchEngineRetrieverMock(opt)


class TestSearchQuerySearchEngineRetriever(unittest.TestCase):
    def setUp(self) -> None:
        parser = ParlaiParser(True, True)
        opt = parser.parse_args(
            [
                '--model',
                'parlai.agents.fid.fid:SearchQuerySearchEngineFiDAgent',
                '--retriever-debug-index',
                'compressed',
            ]
        )
        dictionary = BertTokenizerDictionaryAgent(opt)
        self.rertriever = MockSearchQuerySearchEngineRetriever(opt, dictionary, None)

    def test_retrieval(self):
        retrieved = self.rertriever.retrieve_and_score(
            torch.LongTensor([[1, 2, 3], [10, 20, 0]])
        )
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved, tuple)
        self.assertEqual(len(retrieved), 2)

        retrieved_docs = retrieved[0]
        self.assertIsInstance(retrieved_docs, list)
        self.assertEqual(len(retrieved_docs), 2)

        # With Search query
        second_retrieved_doc = retrieved_docs[0][1]
        self.assertIsInstance(second_retrieved_doc, Document)
        self.assertIsInstance(second_retrieved_doc.get_text(), str)
        self.assertEqual(
            second_retrieved_doc.get_text(), 'content 1 for query " mock search query "'
        )

        # WithOUT Search query
        second_retrieved_doc = retrieved_docs[1][1]
        self.assertIsInstance(second_retrieved_doc, Document)
        self.assertIsInstance(second_retrieved_doc.get_text(), str)
        self.assertEqual(second_retrieved_doc.get_text(), '')


################################################################
# Search query in FAISS index FiD Agent
################################################################


class MockSearchQueryFAISSIndexRetriever(SearchQueryFAISSIndexRetriever):
    def init_search_query_generator(self, opt):
        pass

    def generate_search_query(self, query):
        return ['mock search query']


class TestSearchQueryFAISSIndexRetriever(unittest.TestCase):
    def setUp(self) -> None:
        parser = ParlaiParser(True, True)
        opt = parser.parse_args(
            [
                '--model',
                'parlai.agents.fid.fid:SearchQueryFAISSIndexFiDAgent',
                '--retriever-debug-index',
                'compressed',
            ]
        )
        dictionary = BertTokenizerDictionaryAgent(opt)
        self.rertriever = MockSearchQueryFAISSIndexRetriever(opt, dictionary, None)

    def test_retrieval(self):
        retrieved = self.rertriever.retrieve_and_score(
            torch.LongTensor(np.array([[1, 2, 3]]))
        )
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved, tuple)
        self.assertEqual(len(retrieved), 2)

        retrieved_docs = retrieved[0]
        self.assertIsInstance(retrieved_docs, list)
        self.assertEqual(len(retrieved_docs), 1)

        second_retrieved_doc = retrieved_docs[0][1]
        self.assertIsInstance(second_retrieved_doc, Document)
        self.assertIsInstance(second_retrieved_doc.get_text(), str)
