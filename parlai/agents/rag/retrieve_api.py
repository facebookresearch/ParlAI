#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
APIs for retrieving a list of "Contents" using an "Search Query".

The term "Search Query" here refers to any abstract form of input string. The definition
of "Contents" is also loose and depends on the API.
"""

from abc import ABC, abstractmethod
import requests
from typing import Any, Dict, List

from parlai.core.opt import Opt
from parlai.utils import logging


CONTENT = 'content'
DEFAULT_NUM_TO_RETRIEVE = 5


class RetrieverAPI(ABC):
    """
    Provides the common interfaces for retrievers.

    Every retriever in this modules must implement the `retrieve` method.
    """

    def __init__(self, opt: Opt):
        self.skip_query_token = opt['skip_retrieval_token']

    @abstractmethod
    def retrieve(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        """
        Implements the underlying retrieval mechanism.
        """

    def create_content_dict(self, content: list, **kwargs) -> Dict:
        resp_content = {CONTENT: content}
        resp_content.update(**kwargs)
        return resp_content


class SearchEngineRetrieverMock(RetrieverAPI):
    """
    For unit tests and debugging (does not need a running server).
    """

    def retrieve(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        all_docs = []
        for query in queries:
            if query == self.skip_query_token:
                docs = None
            else:
                docs = []
                for idx in range(num_ret):
                    doc = self.create_content_dict(
                        f'content {idx} for query "{query}"',
                        url=f'url_{idx}',
                        title=f'title_{idx}',
                    )
                    docs.append(doc)
            all_docs.append(docs)
        return all_docs


class SearchEngineRetriever(RetrieverAPI):
    """
    Queries a server (eg, search engine) for a set of documents.

    This module relies on a running HTTP server. For each retrieval it sends the query
    to this server and receives a JSON; it parses the JSON to create the response.
    """

    def __init__(self, opt: Opt):
        super().__init__(opt=opt)
        self.server_address = self._validate_server(opt.get('search_server'))
        self._server_timeout = (
            opt['search_server_timeout']
            if opt.get('search_server_timeout', 0) > 0
            else None
        )
        self._max_num_retries = opt.get('max_num_retries', 0)

    def _query_search_server(self, query_term, n):
        server = self.server_address
        req = {'q': query_term, 'n': n}
        trials = []
        while True:
            try:
                logging.debug(f'sending search request to {server}')
                server_response = requests.post(
                    server, data=req, timeout=self._server_timeout
                )
                resp_status = server_response.status_code
                trials.append(f'Response code: {resp_status}')
                if resp_status == 200:
                    return server_response.json().get('response', None)
            except requests.exceptions.Timeout:
                if len(trials) > self._max_num_retries:
                    break
                trials.append(f'Timeout after {self._server_timeout} seconds.')
        logging.error(
            f'Failed to retrieve data from server after  {len(trials)+1} trials.'
            f'\nFailed responses: {trials}'
        )

    def _validate_server(self, address):
        if not address:
            raise ValueError('Must provide a valid server for search')
        if address.startswith('http://') or address.startswith('https://'):
            return address
        PROTOCOL = 'http://'
        logging.warning(f'No protocol provided, using "{PROTOCOL}"')
        return f'{PROTOCOL}{address}'

    def _retrieve_single(self, search_query: str, num_ret: int):
        if search_query == self.skip_query_token:
            return None

        retrieved_docs = []
        search_server_resp = self._query_search_server(search_query, num_ret)
        if not search_server_resp:
            logging.warning(
                f'Server search did not produce any results for "{search_query}" query.'
                ' returning an empty set of results for this query.'
            )
            return retrieved_docs

        for rd in search_server_resp:
            url = rd.get('url', '')
            title = rd.get('title', '')
            sentences = [s.strip() for s in rd[CONTENT].split('\n') if s and s.strip()]
            retrieved_docs.append(
                self.create_content_dict(url=url, title=title, content=sentences)
            )
        return retrieved_docs

    def retrieve(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        # TODO: update the server (and then this) for batch responses.
        return [self._retrieve_single(q, num_ret) for q in queries]
