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
from .build_tfidf import live_count_matrix_t, get_tfidf_matrix_t
from numpy.random import choice
import math
import os


class SparseTfidfRetrieverAgent(Agent):

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Retriever Arguments')
        parser.add_argument(
            '--sp-retriever-task', type=str, default=None,
            help='ParlAI task to use to "train" retriever')
        parser.add_argument(
            '--sp-retriever-dbpath', type=str, required=True,
            help='/path/to/saved/db.db')
        parser.add_argument(
            '--sp-retriever-tfidfpath', type=str, required=True,
            help='Directory for saving output files')
        parser.add_argument(
            '--sp-retriever-numworkers', type=int, default=None,
            help='Number of CPU processes (for tokenizing, etc)')
        parser.add_argument(
            '--sp-retriever-ngram', type=int, default=2,
            help='Use up to N-size n-grams (e.g. 2 = unigrams + bigrams)')
        parser.add_argument(
            '--sp-retriever-hashsize', type=int, default=int(math.pow(2, 24)),
            help='Number of buckets to use for hashing ngrams')
        parser.add_argument(
            '--sp-retriever-tokenizer', type=str, default='simple',
            help='String option specifying tokenizer type to use '
                 '(e.g. "corenlp")')
        parser.add_argument(
            '--sp-retriever-mode', choices=['keys', 'values'], default='values',
            help='Whether to retrieve the stored key or the stored value.'
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'SparseTfidfRetrieverAgent'

        # we'll need to build the tfid if it's not already
        rebuild_tfidf = not os.path.exists(opt['retriever_tfidfpath'])
        # sets up db
        if not os.path.exists(opt['retriever_dbpath']):
            if not opt.get('retriever_task'):
                opt['retriever_task'] = opt['task']
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
            tfidf_path=opt['retriever_tfidfpath'], strict=False)
        self.cands_hash = {}
        self.ret_mode = opt['retriever_mode']

    def train(mode=True):
        self.training = mode

    def eval():
        self.training = False

    def doc2txt(self, docid):
        if self.ret_mode == 'keys':
            return self.db.get_doc_text(docid)
        elif self.ret_mode == 'values':
            return self.db.get_doc_value(docid)
        else:
            raise RuntimeError('Retrieve mode {} not yet supported.'.format(
                self.ret_mode))

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        if 'text' in obs:
            # good--we should reply
            doc_ids, doc_scores = self.ranker.closest_docs(obs['text'], k=30)

            if obs.get('label_candidates') and False:
                # these are better selection than stored facts
                # rank these options instead
                cands = obs['label_candidates']
                cands_id = id(cands)
                if cands_id not in self.cands_hash:
                    # cache candidate set
                    # will not update if cand set changes contents
                    c_list = list(cands)
                    self.cands_hash[cands_id] = (
                        get_tfidf_matrix_t(
                            live_count_matrix_t(self.tfidf_args, c_list)
                        ),
                        c_list
                    )
                c_ids, c_scores = self.ranker.closest_docs(obs['text'], k=30, matrix=self.cands_hash[cands_id][0])
                reply['text_candidates'] = [self.cands_hash[cands_id][1][cid] for cid in c_ids]
                reply['text'] = reply['text_candidates'][0]
            elif len(doc_ids) > 0:
                # return stored fact
                total = sum(doc_scores)
                doc_probs = [d / total for d in doc_scores]

                # returned
                picks = [self.doc2txt(int(did)) for did in doc_ids]
                reply['text_candidates'] = picks

                # could pick single choice based on probability scores?
                # pick = int(choice(doc_ids, p=doc_probs))
                pick = int(doc_ids[0])  # select best response
                reply['text'] = self.doc2txt(pick)
            else:
                # no cands and nothing found, return generic response
                reply['text'] = choice([
                    'Can you say something more interesting?',
                    'Why are you being so short with me?',
                    'What are you really thinking?',
                    'Can you expand on that?',
                ])


        return reply
