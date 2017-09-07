# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""a string match retriever."""


import copy
import os
import logging
from numpy import random
import sqlite3


from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from .ir_util import (
    DEFAULT_LENGTH_PENALTY,
    MaxPriorityQueue,
    build_query_representation,
    score_match,
    stopwords,
)

class StringMatchRetrieverAgent(Agent):
    """Builds and/or loads a string match retriever

    The retriever identifies all facts that overlap the input query string, and
    output these facts either in a random order, or by frequency decreasing.
    """

    DEFAULT_MAX_FACTS = 100000
    DOC_TABLE_NAME = 'document'
    FREQ_TABLE_NAME = 'freq'

    @staticmethod
    def print_info(msg):
        logging.info("[ StringMatchRetriever ]: " + str(msg))

    @staticmethod
    def add_cmdline_args(argparser):
        retriever = argparser.add_argument_group('Retriever Arguments')
        retriever.add_argument(
            '--retriever-file',
            help='if set, the retriever will save to this path as default',
        )
        retriever.add_argument(
            '--retriever-maxexs',
            default=StringMatchRetrieverAgent.DEFAULT_MAX_FACTS,
            type=int,
            help='max number of examples to build retriever on',
        )

    def __init__(self, opt):
        super().__init__(opt)
        self.id = 'StringMatchRetrieverAgent'
        self.dict_agent = DictionaryAgent(opt)
        self.token2facts = {}
        self.facts = []
        self.length_penalty = float(opt.get('length_penalty') or DEFAULT_LENGTH_PENALTY)
        is_file_exists = os.path.isfile(opt.get('retriever_file'))
        self.sql_connection = sqlite3.connect(opt.get('retriever_file'))
        self.cursor = self.sql_connection.cursor()
        if not is_file_exists:
            self.cursor.execute(
                "CREATE TABLE %s (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, fact)"
                % self.DOC_TABLE_NAME
            )
            self.cursor.execute(
                "CREATE TABLE %s (token, fact_id, freq)"
                % self.FREQ_TABLE_NAME
            )

    def _get_fact_id(self, fact):
        self.cursor.execute(
            "SELECT fact_id FROM %s WHERE fact = ?" % self.DOC_TABLE_NAME,
            (fact,),
        )
        try:
            return int(self.cursor.fetchall()[0][0])
        except ValueError or IndexError:
            return -1

    def act(self):
        fact = self.observation.get('text')
        self.cursor.execute(
            "INSERT INTO %s(fact) VALUES(?)" % self.DOC_TABLE_NAME,
            (fact, ),
        )
        # TODO: the following fact_id assignment won't work for multi-thread
        fact_id = self.cursor.lastrowid
        token_cnt = {}
        for _token in set(self.dict_agent.tokenize(fact.lower())):
            if _token in stopwords:
                continue
            if _token not in token_cnt:
                token_cnt[_token] = 0
            token_cnt[_token] += 1
        for (_token, _cnt) in token_cnt.items():
            self.cursor.execute(
                "INSERT INTO %s(token, fact_id, freq) VALUES(?, ?, ?)" %
                self.FREQ_TABLE_NAME,
                (_token, fact_id, _cnt,),
            )
        return {'id': 'Retriever'}


    def _get_facts_names(self, fact_ids):
        formatted_fact_ids = [int(_fact_id) for _fact_id in fact_ids]
        return [_row[0] for _row in \
            self.cursor.execute(
                "SELECT fact FROM %s WHERE fact_id in (%s)" %
                (self.DOC_TABLE_NAME, ','.join('?' for _ in fact_ids)),
                formatted_fact_ids,
            )]

    def retrieve(self, query, max_results=100, ordered_randomly=False):
        query_tokens = set(self.dict_agent.tokenize(query.lower()))
        # compute query representation
        query_rep = build_query_representation(self, query_tokens, self.dict_agent.freqs())
        # gather the candidate facts
        cand_facts = {}
        for _token in query_tokens:
            for _row in self.cursor.execute(
                            "SELECT fact_id FROM %s WHERE token = ?"
                            % self.FREQ_TABLE_NAME,
                            (_token,),
                        ):
                _fact_id = _row[0]
                if _fact_id not in cand_facts:
                    cand_facts[_fact_id] = []
                cand_facts[_fact_id].append(_token)
        if not cand_facts:
            return []
        max_results = min(max_results, len(cand_facts))
        if ordered_randomly:
            fact_ids = random.choice(list(cand_facts.keys()), max_results, replace=False)
            return self._get_facts_names(fact_ids)
        # ordered by score
        result = MaxPriorityQueue(max_results)
        for (_fact_id, _tokens) in cand_facts.items():
            self.cursor.execute(
                "SELECT fact FROM %s WHERE fact_id=?" % self.DOC_TABLE_NAME,
                (_fact_id,),
            )
            _fact = self.cursor.fetchall()[0][0]
            _score = score_match(
                        query_rep,
                        _tokens,
                        self.dict_agent.tokenize(_fact),
                     )
            result.add(_fact, _score)
        return reversed(result)

    def save(self):
        self.sql_connection.commit()
        self.print_info('model successfully saved.')
