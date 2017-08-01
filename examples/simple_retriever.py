# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
from parlai.core.dict import DictionaryAgent


stopwords = { 'i', 'a', 'an', 'are', 'about', 'as', 'at', 'be', 'by',
              'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or',
              'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where',
              '--', '?', '.', "''", "''", "``", ',', 'do', 'see', 'want',
              'people', 'and', "n't", "me", 'too', 'own', 'their', '*',
              "'s", 'not', 'than', 'other', 'you', 'your', 'know', 'just',
              'but', 'does', 'really', 'have', 'into', 'more', 'also',
              'has', 'any', 'why', 'will', 'to', 'am', 'who'}

rewrites = [
    ['married', 'husband'],
    ['married', 'wife'],
    ['like', 'love', 'enjoy', 'awesome', 'great'],
    ['born', 'birthdate', 'birthday'],
    ['born', 'come from'],
    ['boyfriend', 'dating'],
    ['girlfriend', 'dating'],
    ['work', 'works', 'employed'],
    ['watch', 'see'],
    ['movie', 'film'],
    ['food', 'eat']
]
rewrite_hash = {}

def score_match(query_rep, text, length_penalty, debug=False):
    words = text.lower().split(' ')
    score = 0
    rw = query_rep['words']
    used = {}
    for w in words:
        if w in rw and w not in used:
            score += 1
            if debug:
                print("match: " + w)
        used[w] = True
    norm = math.sqrt(len(used))
    score = score / math.pow(norm * query_rep['norm'], length_penalty)
    return score

def retrieve_candidates(query_rep, cands, length_penalty):
    """ Rank candidates given representation of query """
    cands = list(cands)
    score = [0] * len(cands)
    new_cands = []
    for i, c in enumerate(cands):
        score[i] = score_match(query_rep, c, length_penalty)
        print("\t[ " + c + " " + str(score[i]) + " ]")
        if score[i] > 0:
            new_cands.append(c)
    return new_cands

def build_query_rewrite_hash():
    for i in range(len(rewrites)):
        s = rewrites[i]
        for j in s:
            if j in rewrite_hash:
                rewrite_hash[j] = list(set(s).union(rewrite_hash[j]))
            else:
                rewrite_hash[j] = s
            # print(j, rewrite_hash[j])
            
def build_query_rewrite(query):
    if len(rewrite_hash) == 0:
        build_query_rewrite_hash()
    query = query.replace('?', ' ')
    query = query.replace('  ', ' ')
    words = query.lower().split(' ')
    new_words = set()
    for w in words:
        if w not in stopwords:
            if w in rewrite_hash:
                new_words.update(rewrite_hash[w])
            else:
                new_words.add(w)
    new_query = (' ').join(new_words)
    return new_query

def build_query_representation(query):
    """ Build representation of query, e.g. words or n-grams """
    query = build_query_rewrite(query)
    print("rewrite_query:", query)
    rep = {}
    rep['words'] = {}
    words = query.lower().split(' ')
    rw = rep['words']
    used = {}
    
    for w in words:
        #if len(self.dictionary.freqs()) > 0:
        #    rw[w] = 1.0 / (1.0 + math.log(1.0 + self.dictionary.freqs()[w]))
        #else:
        #    if w not in stopwords:
        #        rw[w] = 1
        if w not in stopwords:
            rw[w] = 1
        used[w] = True
    norm = len(used)
    rep['norm'] = math.sqrt(len(words))
    return rep


def retriever(query, context):
    print("query:", query)
    print("context:", context.replace('\n','|'))
    rep = build_query_representation(query)
    cands = context.split('\n')
    new_cands = retrieve_candidates(rep, cands, 1)
    #rank_candidates(rep, obs['label_candidates'],
    #                self.length_penalty))
    new_context = '\n'.join(new_cands)
    print("new context:", new_context.replace('\n','|'))
    return new_context

query = 'who am i married to?'
context = 'my husband is darren.\ni like goats.'
new_context = retriever(query, context)

