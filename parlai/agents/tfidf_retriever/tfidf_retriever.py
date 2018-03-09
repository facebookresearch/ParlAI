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
from .build_tfidf import live_count_matrix, get_tfidf_matrix
from numpy.random import choice
import math
import os


class TfidfRetrieverAgent(Agent):
    """TFIDF-based retriever agent.

    If given a task to specify, will first store entries of that task into
    a SQLite database and then build a sparse tfidf matrix of those entries.
    If not, will build it on-the-fly whenever it sees observations with labels.

    This agent generates responses by building a sparse entry of the incoming
    text observation, and then returning the highest-scoring documents
    (calculated via sparse matrix multiplication) from the tfidf matrix.

    By default, this will return the "value" (the response) of the closest
    entries. For example, saying "What is your favorite movie?" will not return
    the text "Which movie is your favorite?" (high match) but rather the reply
    to that (e.g. "Jurassic Park"). To use this agent for retrieving texts
    (e.g. Wikipedia Entries), you can store label-less examples with the
    '--retriever-task' argument and switch '--retriever-mode' to 'keys'.
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Retriever Arguments')
        parser.add_argument(
            '--retriever-task', type=str, default=None,
            help='ParlAI task to use to "train" retriever. If not given, no '
                 'entries will initially populate the database / matrix.'
                 'Calling observe/act with a labels will store them.')
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
        parser.add_argument(
            '--retriever-mode', choices=['keys', 'values'], default='values',
            help='Whether to retrieve the stored key or the stored value.'
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'SparseTfidfRetrieverAgent'

        # we'll need to build the tfid if it's not already
        rebuild_tfidf = not os.path.exists(opt['retriever_tfidfpath'] + '.npz')
        # sets up db
        if not os.path.exists(opt['retriever_dbpath']):
            build_db(opt, opt.get('retriever_task'), opt['retriever_dbpath'],
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
        self.ret_mode = opt['retriever_mode']
        self.cands_hash = {}  # cache for candidates
        self.triples_to_add = []  # in case we want to add more entries

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def doc2txt(self, docid):
        if self.ret_mode == 'keys':
            return self.db.get_doc_text(docid)
        elif self.ret_mode == 'values':
            return self.db.get_doc_value(docid)
        else:
            raise RuntimeError('Retrieve mode {} not yet supported.'.format(
                self.ret_mode))

    def store(self, observation):
        tpl = (None, observation['text'], choice(observation['labels']))
        self.triples_to_add.append(tpl)

    def rebuild(self):
        if len(self.triples_to_add) > 0:
            self.db.add(self.triples_to_add)
            self.triples_to_add.clear()
            # rebuild tfidf
            build_tfidf(self.tfidf_args)
            self.ranker = TfidfDocRanker(
                tfidf_path=self.opt['retriever_tfidfpath'], strict=False)

    def save(self, path=None):
        self.rebuild()

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        if 'text' in obs:
            # we should reply or add this as a stored example
            if 'labels' in obs:
                # save this to our retrieval matrix
                self.store(obs)
            else:
                self.rebuild()  # no-op if nothing has been queued to store
                doc_ids, doc_scores = self.ranker.closest_docs(obs['text'], k=30)

                if obs.get('label_candidates'):
                    # these are better selection than stored facts
                    # rank these options instead
                    cands = obs['label_candidates']
                    cands_id = id(cands)
                    if cands_id not in self.cands_hash:
                        # cache candidate set
                        # will not update if cand set changes contents
                        c_list = list(cands)
                        self.cands_hash[cands_id] = (
                            get_tfidf_matrix(
                                live_count_matrix(self.tfidf_args, c_list)
                            ),
                            c_list
                        )
                    c_ids, c_scores = self.ranker.closest_docs(obs['text'], k=30, matrix=self.cands_hash[cands_id][0])
                    reply['text_candidates'] = [self.cands_hash[cands_id][1][cid] for cid in c_ids]
                    reply['text'] = reply['text_candidates'][0]
                elif len(doc_ids) > 0:
                    # return stored fact
                    # total = sum(doc_scores)
                    # doc_probs = [d / total for d in doc_scores]

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
