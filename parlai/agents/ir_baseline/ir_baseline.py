# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Simple IR baselines.
# We plan to implement the following variants:
# Given an input message, either:
# (i) find the most similar message in the (training) dataset and output the response from that exchange; or
# (ii) find the most similar response to the input directly.
# (iii) if label_candidates are provided, simply ranks them according to their similarity to the input message.
# Currently only (iii) is used.
#
# Additonally, TFIDF is either used (requires building a dictionary) or not,
# depending on whether you train on the train set first, or not.

import math
import random
from collections.abc import Sequence
import heapq

from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent

class MaxPriorityQueue(Sequence):
    def __init__(self, max_size):
        self.capacity = max_size
        self.lst = []

    def add(self, item, priority=None):
        if priority is None:
            priority = item
        if len(self.lst) < self.capacity:
            heapq.heappush(self.lst, (priority, item))
        elif priority > self.lst[0][0]:
            heapq.heapreplace(self.lst, (priority, item))

    def __getitem__(self, key):
        return sorted(self.lst)[key][1]

    def __len__(self):
        return len(self.lst)

    def __str__(self):
        return str([v for _, v in sorted(self.lst)])

    def __repr__(self):
        return repr([v for _, v in sorted(self.lst)])


stopwords = { 'i', 'a', 'an', 'are', 'about', 'as', 'at', 'be', 'by',
              'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or',
              'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where',
              '--', '?', '.', "''", "''", "``", ',', 'do', 'see', 'want',
              'people', 'and', "n't", "me", 'too', 'own', 'their', '*',
              "'s", 'not', 'than', 'other', 'you', 'your', 'know', 'just',
              'but', 'does', 'really', 'have', 'into', 'more', 'also',
              'has', 'any', 'why', 'will'}

def score_match(query_rep, text, length_penalty, debug=False, dictionary=None):
    if not dictionary:
       words = text.lower().split(' ')
    else:
       words = [w for w in dictionary.tokenize(text.lower())]
    score = 0
    rw = query_rep['words']
    used = {}
    for w in words:
        if w in rw and w not in used:
            score += rw[w]
            if debug:
                print("match: " + w)
        used[w] = True
    norm = math.sqrt(len(used))
    score = score / math.pow(norm * query_rep['norm'], length_penalty)
    return score

def rank_candidates(query_rep, cands, length_penalty, dictionary=None):
    """ Rank candidates given representation of query """
    if True:
        mpq = MaxPriorityQueue(100)
        for c in cands:
            score = score_match(query_rep, c, length_penalty, dictionary)
            mpq.add(c, score)
        return list(reversed(mpq))
    else:
        cands = list(cands)
        score = [0] * len(cands)
        for i, c in enumerate(cands):
            score[i] = -score_match(query_rep, c, length_penalty, dictionary)
        r = [i[0] for i in sorted(enumerate(score), key=lambda x:x[1])]
        res = []
        for i in range(min(100, len(score))):
            res.append(cands[r[i]])
        print(score[r[0]])
        return res


class IrBaselineAgent(Agent):

    @staticmethod
    def add_cmdline_args(parser):
        DictionaryAgent.add_cmdline_args(parser)
        parser.add_argument(
            '-lp', '--length_penalty', default=0.5,
            help='length penalty for responses')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'IRBaselineAgent'
        self.length_penalty = float(opt['length_penalty'])
        self.dictionary = DictionaryAgent(opt)
        self.opt = opt

    def observe(self, obs):
        self.observation = obs
        self.dictionary.observe(obs)
        return obs

    def act(self):
        if self.opt.get('datatype', '').startswith('train'):
            self.dictionary.act()

        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        # Rank candidates
        if 'label_candidates' in obs and len(obs['label_candidates']) > 0:
            rep = self.build_query_representation(obs['text'])
            reply['text_candidates'] = (
                rank_candidates(rep, obs['label_candidates'],
                                self.length_penalty, self.dictionary))
            reply['text'] = reply['text_candidates'][0]
        else:
            reply['text'] = "I don't know."
        return reply

    def save(self, fname=None):
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            self.dictionary.save(fname + '.dict')

    def load(self, fname):
        self.dictionary.load(fname + '.dict')

    def build_query_representation(self, query):
        """ Build representation of query, e.g. words or n-grams """
        rep = {}
        rep['words'] = {}
        words = [w for w in self.dictionary.tokenize(query.lower())]
        rw = rep['words']
        used = {}
        for w in words:
            if len(self.dictionary.freqs()) > 0:
                rw[w] = 1.0 / (1.0 + math.log(1.0 + self.dictionary.freqs()[w]))
            else:
                if w not in stopwords:
                    rw[w] = 1
            used[w] = True
        norm = len(used)
        rep['norm'] = math.sqrt(len(words))
        return rep
