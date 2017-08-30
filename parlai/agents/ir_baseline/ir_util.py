# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# utils for IR usage.


from collections.abc import Sequence
import heapq
import math


DEFAULT_LENGTH_PENALTY = 0.5

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


def build_query_representation(self, tokens, freqs):
    """ Build representation of query """
    tokens_set = set(tokens)
    rep = {}
    rep_tokens = {}
    rep['tokens'] = rep_tokens
    for token in tokens_set:
        if freqs:
            rep_tokens[token] = 1.0 / (1.0 + math.log(1.0 + freqs.get(token, 0)))
        else:
            if token not in stopwords:
                rep_tokens[token] = 1
    rep['norm'] = math.sqrt(len(tokens_set))
    return rep


def score_match(query_rep, text_tokens, length_penalty=DEFAULT_LENGTH_PENALTY):
    text_tokens_set = set(text_tokens)
    score = 0
    rep_tokens = query_rep['tokens']
    for token in text_tokens_set:
        # check: change from score += 1
        score += rep_tokens[token]
    norm = math.sqrt(len(text_tokens_set))
    score = score / math.pow(norm * query_rep['norm'], length_penalty)
    return score
