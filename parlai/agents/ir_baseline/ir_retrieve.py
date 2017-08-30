# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""a string match retriever."""


import copy
from numpy import random

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

    def __init__(self, opt):
        super().__init__(opt)
        self.id = 'StringMatchRetrieverAgent'
        self.dict_agent = DictionaryAgent(opt)
        self.token2facts = {}
        self.facts = []
        self.length_penalty = float(opt.get('length_penalty') or DEFAULT_LENGTH_PENALTY)

    def act(self):
        fact = self.observation.get('text')
        self.facts.append(fact)
        for token in set(self.dict_agent.tokenize(fact)):
            if token in stopwords:
                continue
            if token not in self.token2facts:
                self.token2facts[token] = []
            self.token2facts[token].append(fact)
        return {'id': 'Retriever'}

    def retrieve(self, query, max_results=100, ordered_randomly=False):
        query_tokens = set(self.dict_agent.tokenize(query))
        # compute query representation
        query_rep = build_query_representation(self, query_tokens, self.dict_agent.freqs())
        # gather the candidate facts
        cand_facts = {}
        for token in query_tokens:
            if token not in self.token2facts:
                continue
            for fact in self.token2facts[token]:
                if fact not in cand_facts:
                    cand_facts[fact] = [token]
                else:
                    cand_facts[fact].append(token)
        if not cand_facts:
            return []
        if ordered_randomly:
            return list(random.choice(list(cand_facts.keys()), max_results, replace=False))
        # ordered by score
        facts_score = {}
        for (fact, tokens) in cand_facts.items():
            facts_score[fact] = score_match(query_rep, tokens)
        max_results = min(max_results, len(cand_facts))
        result = MaxPriorityQueue(max_results)
        for (fact, score) in facts_score.items():
            result.add(fact, score)
        return reversed(result)

    def load(self, filename):
        """Load pre-existing StringMatchRetriever in format:
            the first line: 'fact1<TAB>fact2<TAB>...'
            starting from second line: 'token<TAB>fact_index1<TAB>fact_index2...'
        """

        print('StringMatchRetriever: loading StringMatch from {}'.format(
              filename))
        if not hasattr(self, 'facts'):
            self.facts = []
        if not hasattr(self, 'token2facts'):
            self.token2facts = {}
        with open(filename) as read:
            for line in read.readlines():
                if not self.facts:
                    # the first line contains all the facts
                    self.facts = line.strip().split('\t')
                    if not self.facts:
                        print("StringMatch: Empty model loaded.")
                        return
                    continue
                split = line.strip().split('\t')
                token = split[0]
                self.token2facts[token] = [
                    self.facts[int(_fact_ind)] for _fact_ind in split[1:]
                ]
        print('StringMatchRetriever: %d facts with %d tokens loaded.'
              % (len(self.facts), len(self.token2facts)))

    def save(self, filename=None):
        """Save StringMatchRetriever to file.
        Format is:
            the first line: 'fact1<TAB>fact2<TAB>...'
            starting from second line: 'token<TAB>fact_index1<TAB>fact_index2...'
        """
        filename = self.opt['model_file'] if filename is None else filename
        print('StringMatchRetriever: saving model to {}'.format(filename))
        with open(filename, 'w') as write:
            write.write('\t'.join(self.facts) + '\n')
            facts_ind = {fact: ind for (ind, fact) in enumerate(self.facts)}
            for token in self.token2facts:
                write.write(
                    token + '\t'
                    + '\t'.join([str(facts_ind[fact]) for fact in self.token2facts[token]])
                    + '\n'
                )
            write.close()
        print('StringMatchRetriever: model successfully saved.')
