#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Simple IR baselines.

We plan to implement the following variants:
Given an input message, either:
(i) find the most similar message in the (training) dataset and output the
  response from that exchange; or
(ii) find the most similar response to the input directly.
(iii) if label_candidates are provided, simply ranks them according to their
  similarity to the input message.

Currently only (iii) is used.

Additonally, TFIDF is either used (requires building a dictionary) or not,
depending on whether you train on the train set first, or not.
"""

import math
from collections.abc import Sequence
import heapq
import json
import torch

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent


class MaxPriorityQueue(Sequence):
    """
    Fixed-size priority queue keeping the max_size largest items.
    """

    def __init__(self, max_size):
        """
        Initialize priority queue.

        :param max_size: maximum capacity of priority queue.
        """
        self.capacity = max_size
        self.lst = []

    def add(self, item, priority=None):
        """
        Add element to the queue, with a separate priority if desired.

        Element will not be added if the queue is at capacity and the element
        has lower priority than the lowest currently in the queue.

        :param item: item to add to queue.
        :param priority: priority to use for item. if None (default), will use
                         the item itself to calculate its own priority.
        """
        if priority is None:
            priority = item
        if len(self.lst) < self.capacity:
            heapq.heappush(self.lst, (priority, item))
        elif priority > self.lst[0][0]:
            heapq.heapreplace(self.lst, (priority, item))

    def __getitem__(self, key):
        """
        Get item at specified index.

        :param key: integer index into priority queue, 0 <= index < max_size.

        :returns: item stored at the specified index.
        """
        return sorted(self.lst)[key][1]

    def __len__(self):
        """
        Return length of priority queue.
        """
        return len(self.lst)

    def __str__(self):
        """
        Return str representation of the priority queue in list form.
        """
        return str([v for _, v in sorted(self.lst)])

    def __repr__(self):
        """
        Return repr representation of the priority queue in list form.
        """
        return repr([v for _, v in sorted(self.lst)])


stopwords = {
    'i',
    'a',
    'an',
    'are',
    'about',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'how',
    'in',
    'is',
    'it',
    'of',
    'on',
    'or',
    'that',
    'the',
    'this',
    'to',
    'was',
    'what',
    'when',
    'where',
    '--',
    '?',
    '.',
    "''",
    "''",
    "``",
    ',',
    'do',
    'see',
    'want',
    'people',
    'and',
    "n't",
    "me",
    'too',
    'own',
    'their',
    '*',
    "'s",
    'not',
    'than',
    'other',
    'you',
    'your',
    'know',
    'just',
    'but',
    'does',
    'really',
    'have',
    'into',
    'more',
    'also',
    'has',
    'any',
    'why',
    'will',
}


def score_match(query_rep, text, length_penalty, dictionary=None, debug=False):
    """
    Calculate the score match between the query representation the text.

    :param query_rep: base query representation to match text again.
    :param text: string to comapre against query_rep for matching tokens
    :param length_penalty: scores are divided by the norm taken to this power
    :dictionary: optional dictionary to use to tokenize text
    :debug: flag to enable printing every match

    :returns: float score of match
    """
    if text == "":
        return 0
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
    norm = math.pow(norm * query_rep['norm'], length_penalty)
    if norm > 1:
        score /= norm
    return score


def rank_candidates(query_rep, cands, length_penalty, dictionary=None):
    """
    Rank candidates given representation of query.

    :param query_rep: base query representation to match text again.
    :param cands: strings to compare against query_rep for matching tokens
    :param length_penalty: scores are divided by the norm taken to this power
    :dictionary: optional dictionary to use to tokenize text

    :returns: ordered list of candidate strings in score-ranked order
    """
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
        r = [i[0] for i in sorted(enumerate(score), key=lambda x: x[1])]
        res = []
        for i in range(min(100, len(score))):
            res.append(cands[r[i]])
        return res


class IrBaselineAgent(Agent):
    """
    Information Retrieval baseline.
    """

    @staticmethod
    def add_cmdline_args(parser):
        """
        Add command line args specific to this agent.
        """
        parser = parser.add_argument_group('IrBaseline Arguments')
        parser.add_argument(
            '-lp',
            '--length_penalty',
            type=float,
            default=0.5,
            help='length penalty for responses',
        )
        parser.add_argument(
            '-hsz',
            '--history_size',
            type=int,
            default=1,
            help='number of utterances from the dialogue history to take use '
            'as the query',
        )
        parser.add_argument(
            '--label_candidates_file',
            type=str,
            default=None,
            help='file of candidate responses to choose from',
        )

    def __init__(self, opt, shared=None):
        """
        Initialize agent.
        """
        super().__init__(opt)
        self.id = 'IRBaselineAgent'
        self.length_penalty = float(opt['length_penalty'])
        self.dictionary = DictionaryAgent(opt)
        self.opt = opt
        self.history = []
        self.episodeDone = True
        if opt.get('label_candidates_file'):
            f = open(opt.get('label_candidates_file'))
            self.label_candidates = f.read().split('\n')

    def reset(self):
        """
        Reset agent properties.
        """
        self.observation = None
        self.history = []
        self.episodeDone = True

    def observe(self, obs):
        """
        Store and remember incoming observation message dict.
        """
        self.observation = obs
        self.dictionary.observe(obs)
        if self.episodeDone:
            self.history = []
        if 'text' in obs:
            self.history.append(obs.get('text', ''))
        self.episodeDone = obs.get('episode_done', False)
        return obs

    def act(self):
        """
        Generate a response to the previously seen observation(s).
        """
        if self.opt.get('datatype', '').startswith('train'):
            self.dictionary.act()

        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        # Rank candidates
        cands = None
        if obs.get('label_candidates', False) and len(obs['label_candidates']) > 0:
            cands = obs['label_candidates']
        if hasattr(self, 'label_candidates'):
            # override label candidates with candidate file if set
            cands = self.label_candidates
        if cands:
            hist_sz = self.opt.get('history_size', 1)
            left_idx = max(0, len(self.history) - hist_sz)
            text = ' '.join(self.history[left_idx : len(self.history)])
            rep = self.build_query_representation(text)
            reply['text_candidates'] = rank_candidates(
                rep, cands, self.length_penalty, self.dictionary
            )
            reply['text'] = reply['text_candidates'][0]
        else:
            reply['text'] = "I don't know."
        return reply

    def save(self, path=None):
        """
        Save dictionary tokenizer if available.
        """
        path = self.opt.get('model_file', None) if path is None else path
        if path:
            self.dictionary.save(path + '.dict')
            data = {}
            data['opt'] = self.opt
            with open(path, 'wb') as handle:
                torch.save(data, handle)
            with open(path + '.opt', 'w') as handle:
                json.dump(self.opt, handle)

    def load(self, fname):
        """
        Load internal dictionary.
        """
        self.dictionary.load(fname + '.dict')

    def build_query_representation(self, query):
        """
        Build representation of query, e.g. words or n-grams.

        :param query: string to represent.

        :returns: dictionary containing 'words' dictionary (token => frequency)
                  and 'norm' float (square root of the number of tokens)
        """
        rep = {}
        rep['words'] = {}
        words = [w for w in self.dictionary.tokenize(query.lower())]
        rw = rep['words']
        used = {}
        for w in words:
            if len(self.dictionary.freq) > 0:
                rw[w] = 1.0 / (1.0 + math.log(1.0 + self.dictionary.freq[w]))
            else:
                if w not in stopwords:
                    rw[w] = 1
            used[w] = True
        rep['norm'] = math.sqrt(len(words))
        return rep
