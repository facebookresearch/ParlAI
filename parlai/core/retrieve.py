# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""a string match retreve dictionary."""


from collections import defaultdict
import copy
from numpy import random

# importing DictionaryAgent for accessing the tokenizer,
# TODO: remove this import once "tokenizer" moved to more genearl place
from .agents import Agent
from .dict import DictionaryAgent


class StringMatchRetrieverAgent(Agent):


    def __init__(self, opt):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None
        # TODO: remove usage of DictionaryAgent
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

    def retrive(self, target_str, max_results=100, ordered_by_freq=False):
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
        max_results = min(max_results, len(facts_freq))
        if ordered_by_freq:
            return sorted(facts_freq, key=facts_freq.get, reverse=True)[:max_results]
        else:
            return list(random.choice(list(facts_freq.keys()), max_results, replace=False))
