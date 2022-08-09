#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Union

from parlai.agents.rag.retrieve_api import (
    SearchEngineRetriever,
    SearchEngineRetrieverMock,
)
from parlai.agents.rag.retrievers import Document
from parlai.core.agents import Agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.message import Message


SERVERS = {
    'default': 'RELEVANT_SEARCH_SERVER',
    'test': 'http://test_api',
}


class SearchAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            '--server',
            type=str,
            choices=SERVERS.keys(),
            default='default',
            help='Which search server to use',
        )
        parser.add_argument(
            '--raw-server',
            type=str,
            default=None,
            help='Specify to override the server choices with your own.',
        )
        parser.add_argument(
            '--n-docs', type=int, default=5, help='How many docs to retrieve'
        )
        parser.add_argument(
            '--intra-doc-delimiter',
            type=str,
            default='\n',
            help='How to delimit intra-document contents',
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared=shared)
        opt['search_server'] = SERVERS[opt['server']]
        if opt.get('raw_server') is not None:
            opt['search_server'] = opt['raw_server']
        opt['skip_retrieval_token'] = ''
        self.intra_doc_delim = opt['intra_doc_delimiter']
        self.n_docs = opt['n_docs']
        if shared is None:
            self.search_client = (
                SearchEngineRetriever(opt)
                if opt['server'] != 'test'
                else SearchEngineRetrieverMock(opt)
            )
        else:
            self.search_client = shared['client']
        self.top_docs = []

    def reset(self):
        super().reset()
        self.top_docs = []

    def share(self) -> Dict[str, Any]:
        shared = super().share()
        shared['client'] = self.search_client
        return shared

    def act(self):
        observation = self.observation
        results = self.search_client.retrieve([observation['text']], self.n_docs)[0]
        documents: List[Document] = []
        for doc in results:
            content = (
                self.intra_doc_delim.join(doc['content'])
                if isinstance(doc['content'], list)
                else doc['content']
            )
            documents.append(
                Document(docid=doc['url'], text=content, title=doc['title'])
            )
        reply = {
            'text': '\n'.join([str(doc) for doc in documents]),
            'top_docs': documents,
        }
        return reply

    def respond(
        self, text_or_message: Union[str, Message], **other_message_fields
    ) -> str:
        """
        Override Agent.respond to set top_docs.
        """
        if isinstance(text_or_message, str):
            observation = Message(text=text_or_message, **other_message_fields)
        else:
            observation = Message(**text_or_message, **other_message_fields)
            if 'text' not in observation:
                raise RuntimeError('The agent needs a \'text\' field in the message.')

        if 'episode_done' not in observation:
            observation['episode_done'] = True
        agent = self.clone()
        agent.observe(observation)
        response = agent.act()
        self.top_docs = response['top_docs']
        return response['text']
