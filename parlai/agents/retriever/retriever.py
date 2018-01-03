# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.utils import AttrDict
from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .build_db import store_contents as build_db
from .build_tfidf import run as build_tfidf
from .build_tfidf import live_count_matrix
from numpy.random import choice
import math
import os


class RetrieverAgent(Agent):

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Retriever Arguments')
        parser.add_argument(
            '--retriever-task', type=str, default=None,
            help='ParlAI task to use to "train" retriever')
        parser.add_argument(
            '--retriever-dbpath', type=str, required=True,
            help='/path/to/saved/db.db')
        parser.add_argument(
            '--retriever-tfidfpath', type=str, required=True,
            help='Directory for saving output files')
        parser.add_argument(
            '--retriever-numworkers', type=int, default=None,
            help='Number of CPU processes (for tokenizing, etc)')
        parser.add_argument(
            '--retriever-ngram', type=int, default=2,
            help='Use up to N-size n-grams (e.g. 2 = unigrams + bigrams)')
        parser.add_argument(
            '--retriever-hashsize', type=int, default=int(math.pow(2, 24)),
            help='Number of buckets to use for hashing ngrams')
        parser.add_argument(
            '--retriever-tokenizer', type=str, default='simple',
            help='String option specifying tokenizer type to use '
                 '(e.g. "corenlp")')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RetrieverAgent'

        # we'll need to build the tfid if it's not already
        rebuild_tfidf = not os.path.exists(opt['retriever_tfidfpath'] + '.npz')
        # sets up db
        if not os.path.exists(opt['retriever_dbpath']):
            if not opt.get('retriever_task'):
                raise RuntimeError('Retriever task param required to build db')
            build_db(opt, opt['retriever_task'], opt['retriever_dbpath'],
                     context_length=opt.get('context_length', -1),
                     include_labels=opt.get('include_labels', True))
            # we rebuilt the db, so need to force rebuilding of tfidf
            rebuild_tfidf = True

        self.tfidf_args = AttrDict({
            'db_path': opt['retriever_dbpath'],
            'out_dir': opt['retriever_tfidfpath'],
            'ngram': opt['retriever_ngram'],
            'hash_size': opt['retriever_hashsize'],
            'tokenizer': opt['retriever_tokenizer'],
            'num_workers': opt['retriever_numworkers'],
        })

        if rebuild_tfidf:
            # build tfidf if we built the db or if it doesn't exist
            build_tfidf(self.tfidf_args)

        self.db = DocDB(db_path=opt['retriever_dbpath'])
        self.ranker = TfidfDocRanker(
            tfidf_path=opt['retriever_tfidfpath'] + '.npz', strict=False)
        self.cands_hash = {}

    def train(mode=True):
        self.training = mode

    def eval():
        self.training = False

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        if 'text' in obs:
            doc_ids, doc_scores = self.ranker.closest_docs(obs['text'], k=30)
            if len(doc_ids) == 0:
                reply['text'] = choice([
                    'Can you say something more interesting?',
                    'Why are you being so short with me?',
                    'What are you really thinking?',
                    'Can you expand on that?',
                ])
            else:
                total = sum(doc_scores)
                doc_probs = [d / total for d in doc_scores]

                # rank candidates
                if obs.get('label_candidates'):
                    # these are better selection than stored facts
                    cands = obs['label_candidates']
                    cands_id = id(cands)
                    if cands_id not in self.cands_hash:
                        # cache candidate set
                        # will not update if cand set changes contents
                        c_list = list(cands)
                        self.cands_hash[cands_id] = (live_count_matrix(self.tfidf_args, c_list), c_list)
                    c_ids, c_scores = self.ranker.closest_docs(obs['text'], k=30, matrix=self.cands_hash[cands_id][0])
                    picks = [self.cands_hash[cands_id][1][cid] for cid in c_ids]
                else:
                    picks = [self.db.get_doc_value(int(did)) for did in doc_ids]
                reply['text_candidates'] = picks

                # pick single choice
                pick = int(choice(doc_ids, p=doc_probs))
                reply['text'] = self.db.get_doc_value(pick)

        return reply

# def shorten_text(text):
#     idx = text.rfind('.', 10, 125)
#     if idx > 0:
#         text = text[:idx + 1]
#     else:
#         idx = text.rfind('?', 10, 125)
#         if idx > 0:
#             text = text[:idx + 1]
#         else:
#             idx = text.rfind('!', 10, 125)
#             if idx > 0:
#                 text = text[:idx + 1]
#             else:
#                 idx = text.rfind(' ', 0, 75)
#                 if idx > 0:
#                     text = text[:idx]
#                 else:
#                     text = text[:50]
#     return text
