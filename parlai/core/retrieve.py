# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""a string match retrever."""


from collections import defaultdict
import copy
from numpy import random

from .agents import Agent
# importing DictionaryAgent only for accessing the tokenizer.
from .dict import DictionaryAgent


class StringMatchRetrieverAgent(Agent):
    """Builds and/or loads a string match retriever

    The retriever identifies all facts that overlap the input query string, and
    output these facts either in a random order, or by frequency decreasing.
    """

    def __init__(self, opt):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None
        self.dict_agent = DictionaryAgent(opt)
        self.token2facts = {}
        self.facts = []

    def act(self):
        fact = self.observation.get('text')
        self.facts.append(fact)
        for token in set(self.dict_agent.tokenize(fact)):
            if token not in self.token2facts:
                self.token2facts[token] = []
            self.token2facts[token].append(fact)
        return {'id': 'Retriever'}

    def retrieve(self, target_str, max_results=100, ordered_by_freq=False):
        facts_freq = {}
        for token in set(self.dict_agent.tokenize(target_str)):
            if token not in self.token2facts:
                continue
            token_freq = len(self.token2facts[token])
            for fact in self.token2facts[token]:
                if fact not in facts_freq:
                    facts_freq[fact] = 0
                # freq of a fact = sum of freq of each token
                facts_freq[fact] += token_freq
        if not facts_freq:
            return []
        max_results = min(max_results, len(facts_freq))
        if ordered_by_freq:
            return sorted(facts_freq, key=facts_freq.get, reverse=True)[:max_results]
        else:
            return list(random.choice(list(facts_freq.keys()), max_results, replace=False))
