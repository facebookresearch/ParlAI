#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import regex  # noqa: F401
import scipy  # noqa: F401
import sklearn  # noqa: F401
import unicodedata  # noqa: F401

from parlai.core.agents import Agent
from parlai.utils.io import PathManager
from parlai.utils.misc import AttrDict
from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .build_tfidf import run as build_tfidf
from collections import deque
import math
import random
import os
import json
import sqlite3


class TfidfRetrieverAgent(Agent):
    """
    TFIDF-based retriever agent.

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

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = parser.add_argument_group('Retriever Arguments')
        parser.add_argument(
            '--retriever-numworkers',
            type=int,
            default=None,
            help='Number of CPU processes (for tokenizing, etc)',
        )
        parser.add_argument(
            '--retriever-ngram',
            type=int,
            default=2,
            help='Use up to N-size n-grams (e.g. 2 = unigrams + bigrams)',
        )
        parser.add_argument(
            '--retriever-hashsize',
            type=int,
            default=int(math.pow(2, 24)),
            help='Number of buckets to use for hashing ngrams',
        )
        parser.add_argument(
            '--retriever-tokenizer',
            type=str,
            default='simple',
            help='String option specifying tokenizer type to use.',
        )
        parser.add_argument(
            '--retriever-num-retrieved',
            default=5,
            type=int,
            help='How many docs to retrieve.',
        )
        parser.add_argument(
            '--retriever-mode',
            choices=['keys', 'values'],
            default='values',
            help='Whether to retrieve the stored key or the stored value.',
        )
        parser.add_argument(
            '--remove-title',
            type='bool',
            default=False,
            help='Whether to remove the title from the retrieved passage',
        )
        parser.add_argument(
            '--retriever-mode',
            choices=['keys', 'values'],
            default='values',
            help='Whether to retrieve the stored key or the stored value. For '
            'example, if you want to return the text of an example, use '
            'keys here; if you want to return the label, use values here.',
        )
        parser.add_argument(
            '--index-by-int-id',
            type='bool',
            default=True,
            help='Whether to index into database by doc id as an integer. This \
                  defaults to true for DBs built using ParlAI; for the DrQA \
                  wiki dump, it is necessary to set this to False to \
                  index into the DB appropriately',
        )
        parser.add_argument(
            '--tfidf-context-length',
            default=-1,
            type=int,
            help='Number of past utterances to remember when '
            'building flattened batches of data in multi-'
            'example episodes.',
        )
        parser.add_argument(
            '--tfidf-include-labels',
            default=True,
            type='bool',
            help='Specifies whether or not to include labels '
            'as past utterances when building flattened '
            'batches of data in multi-example episodes.',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'SparseTfidfRetrieverAgent'
        if not opt.get('model_file') or opt['model_file'] == '':
            raise RuntimeError('Must set --model_file')

        opt['retriever_dbpath'] = opt['model_file'] + '.db'
        opt['retriever_tfidfpath'] = opt['model_file'] + '.tfidf'

        self.db_path = opt['retriever_dbpath']
        self.tfidf_path = opt['retriever_tfidfpath']

        self.tfidf_args = AttrDict(
            {
                'db_path': opt['retriever_dbpath'],
                'out_dir': opt['retriever_tfidfpath'],
                'ngram': opt['retriever_ngram'],
                'hash_size': opt['retriever_hashsize'],
                'tokenizer': opt['retriever_tokenizer'],
                'num_workers': opt['retriever_numworkers'],
            }
        )

        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                'CREATE TABLE documents ' '(id INTEGER PRIMARY KEY, text, value);'
            )
            conn.commit()
            conn.close()

        self.db = DocDB(db_path=opt['retriever_dbpath'])
        if os.path.exists(self.tfidf_path + '.npz'):
            if shared is None:
                self.ranker = TfidfDocRanker(
                    tfidf_path=opt['retriever_tfidfpath'], strict=False
                )
            else:
                self.ranker = shared['doc_ranker']
        self.ret_mode = opt['retriever_mode']
        self.cands_hash = {}  # cache for candidates
        self.triples_to_add = []  # in case we want to add more entries

        clen = opt.get('tfidf_context_length', -1)
        self.context_length = clen if clen >= 0 else None
        self.include_labels = opt.get('tfidf_include_labels', True)
        self.reset()

    def share(self):
        shared = super().share()
        shared['doc_ranker'] = self.ranker
        return shared

    def reset(self):
        super().reset()
        self.episode_done = False
        self.current = []
        self.context = deque(maxlen=self.context_length)

    def doc2txt(self, docid):
        if not self.opt.get('index_by_int_id', True):
            docid = self.ranker.get_doc_id(docid)
        if self.ret_mode == 'keys':
            return self.db.get_doc_text(docid)
        elif self.ret_mode == 'values':
            return self.db.get_doc_value(docid)
        else:
            raise RuntimeError(
                'Retrieve mode {} not yet supported.'.format(self.ret_mode)
            )

    def rebuild(self):
        if len(self.triples_to_add) > 0:
            self.db.add(self.triples_to_add)
            self.triples_to_add.clear()
            # rebuild tfidf
            build_tfidf(self.tfidf_args)
            self.ranker = TfidfDocRanker(tfidf_path=self.tfidf_path, strict=False)

    def save(self, path=None):
        self.rebuild()
        with PathManager.open(self.opt['model_file'] + '.opt', 'w') as handle:
            json.dump(self.opt, handle)
        with PathManager.open(self.opt['model_file'], 'w') as f:
            f.write('\n')

    def train_act(self):
        if (
            'ordered' not in self.opt.get('datatype', 'train:ordered')
            or self.opt.get('batchsize', 1) != 1
            or self.opt.get('num_epochs', 1) != 1
        ):
            raise RuntimeError(
                'Need to set --batchsize 1, --datatype train:ordered, --num_epochs 1'
            )
        obs = self.observation
        self.current.append(obs)
        self.episode_done = obs.get('episode_done', False)

        if self.episode_done:
            for ex in self.current:
                if 'text' in ex:
                    text = ex['text']
                    self.context.append(text)
                    if len(self.context) > 1:
                        text = '\n'.join(self.context)

                # add labels to context
                labels = ex.get('labels', ex.get('eval_labels'))
                label = None
                if labels is not None:
                    label = random.choice(labels)
                    if self.include_labels:
                        self.context.append(label)
                # use None for ID to auto-assign doc ids--we don't need to
                # ever reverse-lookup them
                self.triples_to_add.append((None, text, label))

            self.episode_done = False
            self.current.clear()
            self.context.clear()

        return {'id': self.getID(), 'text': obs.get('labels', ['I don\'t know'])[0]}

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()
        if 'labels' in obs:
            return self.train_act()
        if 'text' in obs:
            self.rebuild()  # no-op if nothing has been queued to store
            doc_ids, doc_scores = self.ranker.closest_docs(
                obs['text'], self.opt.get('retriever_num_retrieved', 5)
            )
            if len(doc_ids) > 0:
                picks = [self.doc2txt(int(did)) for did in doc_ids]
                pick = self.doc2txt(int(doc_ids[0]))  # select best response

                if self.opt.get('remove_title', False):
                    picks = ['\n'.join(p.split('\n')[1:]) for p in picks]
                    pick = '\n'.join(pick.split('\n')[1:])
                reply['text_candidates'] = picks
                reply['candidate_scores'] = doc_scores.tolist()

                reply['text'] = pick
                reply['candidate_ids'] = doc_ids.tolist()

        return reply
