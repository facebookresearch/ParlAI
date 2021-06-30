#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
APIs for retrieving a list of "Contents" using an "Search Query"

The term "Search Query" here refers to any abstract form of input string.
The definition of "Contents" is also loose and depends on the API.
"""

from abc import ABC, abstractmethod
import requests
from typing import Any, Dict, List

from parlai.core.opt import Opt
from parlai.utils import logging


CONTENT = 'content'
DEFAULT_NUM_TO_RETRIEVE = 5


class RetrieverAgent(ABC):
    def __init__(self, opt: Opt):
        pass

    @abstractmethod
    def retriev(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        """
        Implementats the underlying retrieval mechanism
        """

    def create_content_dict(self, content: list, **kwargs) -> Dict:
        resp_content = {CONTENT: content}
        resp_content.update(**kwargs)
        return resp_content


class SearchEngineRetrieverMock(RetrieverAgent):
    def retriev(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        all_docs = []
        for query in queries:
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


class SearchEngineRetriever(RetrieverAgent):
    def __init__(self, opt: Opt):
        self.server_address = self._validate_server(opt.get('search_server'))

    def _query_search_server(self, query_term, n):
        server = self.server_address
        req = {'q': query_term, 'n': n}
        logging.debug(f'sending search request to {server}')
        server_response = requests.post(server, data=req)
        resp_status = server_response.status_code
        if resp_status == 200:
            return server_response.json().get('response', None)
        logging.error(
            f'Failed to retrieve data from server! Search server returned status {resp_status}'
        )

    def _validate_server(self, address):
        if not address:
            raise ValueError('Must provide a valid server for search')
        if address.startswith('http://') or address.startswith('https://'):
            return address
        PROTOCOL = 'http://'
        logging.warning(f'No portocol provided, using "{PROTOCOL}"')
        return f'{PROTOCOL}{address}'

    def retriev(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        pass
