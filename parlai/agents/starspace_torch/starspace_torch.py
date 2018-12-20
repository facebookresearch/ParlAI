#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Simple implementation of the starspace algorithm, slightly adapted for dialogue.
# See: https://arxiv.org/abs/1709.03856
# TODO: move this over to TorchRankerAgent when it is ready.

from parlai.core.dict import DictionaryAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_agent import Output
from .modules import Starspace

import torch
from torch import optim
import torch.nn as nn
import copy
import os
import random
import json


class StarspaceTorchAgent(TorchRankerAgent):
    """Simple implementation of the starspace algorithm: https://arxiv.org/abs/1709.03856
    """

    OPTIM_OPTS = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
    }

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        TorchRankerAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('StarSpace Arguments')
        agent.add_argument(
            '-esz', '--embeddingsize', type=int, default=128,
            help='size of the token embeddings')
        agent.add_argument(
            '-enorm', '--embeddingnorm', type=float, default=10,
            help='max norm of word embeddings')
        agent.add_argument(
            '-shareEmb', '--share-embeddings', type='bool', default=True,
            help='whether LHS and RHS share embeddings')
        agent.add_argument(
            '--lins', default=0, type=int,
            help='If set to 1, add a linear layer between lhs and rhs.')
        agent.add_argument(
            '-lr', '--learningrate', type=float, default=0.1,
            help='learning rate')
        agent.add_argument(
            '-margin', '--margin', type=float, default=0.1,
            help='margin')
        agent.add_argument(
            '--input_dropout', type=float, default=0,
            help='fraction of input/output features to dropout during training')
        agent.add_argument(
            '-opt', '--optimizer', default='sgd',
            choices=StarspaceTorchAgent.OPTIM_OPTS.keys(),
            help='Choose between pytorch optimizers. '
                 'Any member of torch.optim is valid and will '
                 'be used with default params except learning '
                 'rate (as specified by -lr).')
        agent.add_argument(
            '-tr', '--truncate', default=1000, type=int,
            help='Truncate input lengths to increase speed / '
                 'use less memory.')
        agent.add_argument(
            '-k', '--neg-samples', type=int, default=10,
            help='number k of negative samples per example')
        agent.add_argument(
            '--parrot-neg', type=bool, default=False,
            help='include query as a negative')
        agent.add_argument(
            '--tfidf', type='bool', default=False,
            help='Use frequency based normalization for embeddings.')
        agent.add_argument(
            '-cs', '--cache-size', type=int, default=1000,
            help='size of negative sample cache to draw from')
        agent.add_argument(
            '-cands', '--candidates', type=str, default='custom',
            choices=['batch', 'inline', 'fixed', 'vocab'],
            help='The source of candidates during training '
                 '(see TorchRankerAgent._build_candidates() for details).')
        agent.add_argument(
            '-ecands', '--eval-candidates', type=str, default='inline',
            choices=['batch', 'inline', 'fixed', 'vocab', 'custom'],
            help='The source of candidates during evaluation (defaults to the same'
                 'value as --candidates if no flag is given)')
        StarspaceTorchAgent.dictionary_class().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt
        self.reset_metrics()
        self.id = 'Starspace'
        self.NULL_IDX = 0
        self.cands = torch.LongTensor(1, 1, 1)
        self.ys_cache = []
        self.ys_cache_sz = opt['cache_size']
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        if shared:
            torch.set_num_threads(1)
            # set up shared properties
            self.dict = shared['dict']
        else:
            self.build_model()

        # set up criterion
        self.criterion = torch.nn.CosineEmbeddingLoss(
            margin=opt['margin'], size_average=False
        )
        self.reset()

    def build_model(self):
        print("[ creating StarspaceTorchAgent ]")
        # this is not a shared instance of this class, so do full init
        if (self.opt.get('model_file') and
                (os.path.isfile(self.opt.get('model_file') + '.dict')
                 or (self.opt['dict_file'] is None))):
            # set default dict-file if not set
            self.opt['dict_file'] = self.opt['model_file'] + '.dict'
        # load dictionary and basic tokens & vectors
        self.dict = DictionaryAgent(self.opt)

        self.model = Starspace(self.opt, len(self.dict), self.dict)
        if self.opt.get('model_file') and os.path.isfile(self.opt['model_file']):
            self.load(self.opt['model_file'])
        else:
            self._init_embeddings()
        self.model.share_memory()
        if self.use_cuda:
            self.model.cuda()

    def train_step(self, batch):
        """Train on a single batch of examples."""
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)
        self.model.train()
        self.optimizer.zero_grad()

        if not batch.candidates:
            print('No negatives yet!')
            return Output()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.opt['candidates'], mode='train')

        if self.opt.get('input_dropout', 0) > 0:
            # TODO: fix this input dropout function!!!!!!
            batch, cand_vecs = self.input_dropout(
                batch.text_vec,
                batch.label_vec,
                cand_vecs
            )

        xe, ye = self.model(
            batch.text_vec,
            ys=batch.label_vec,
            cands=cand_vecs
        )

        xe_cat = torch.cat([xe.unsqueeze(1)] * ye.size(1), dim=1)
        y = -torch.ones(xe_cat.size(0), xe_cat.size(1))
        y_0 = torch.ones(xe_cat.size(0))
        y[:, 0] = y_0
        if self.use_cuda:
            y = y.cuda()

        loss = self.criterion(
            xe_cat.view(-1, xe_cat.size(-1)),
            ye.view(-1, xe_cat.size(-1)),
            y.view(-1)
        )
        loss.backward()
        self.optimizer.step()
        scores = nn.CosineSimilarity(dim=-1).forward(xe_cat, ye)

        # Update metrics
        self.metrics['loss'] += loss.item()
        self.metrics['examples'] += batchsize
        _, ranks = scores.sort(1, descending=True)
        for b in range(batchsize):
            rank = (ranks[b] == label_inds[b]).nonzero().item()
            self.metrics['rank'] += 1 + rank

        # Get predictions but not full rankings for the sake of speed
        for i in range(batchsize):
            cands[i] = [batch.labels[i]] + cands[i]

        if cand_vecs.dim() == 2:
            preds = [cands[ordering[0]] for ordering in ranks]
        elif cand_vecs.dim() == 3:
            preds = [cands[i][ordering[0]] for i, ordering in enumerate(ranks)]

        # TODO: this is kinda wrong? this isn't really the label candidate ranking
        # but it will tell us the accuracy
        return Output(preds)

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        batchsize = batch.text_vec.size(0)
        self.model.eval()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.opt['eval_candidates'], mode='eval')

        xe, ye = self.model(
            batch.text_vec,
            ys=None,
            cands=cand_vecs
        )

        xe_cat = torch.cat([xe.unsqueeze(1)] * ye.size(1), dim=1)
        scores = nn.CosineSimilarity(dim=-1).forward(xe_cat, ye)
        _, ranks = scores.sort(1, descending=True)

        # Update metrics
        if label_inds is not None:
            loss = self.rank_loss(scores, label_inds)
            self.metrics['loss'] += loss.item()
            self.metrics['examples'] += batchsize
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero().item()
                self.metrics['rank'] += 1 + rank

        cand_preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:
                cand_list = cands
            elif cand_vecs.dim() == 3:
                cand_list = cands[i]
            cand_preds.append([cand_list[rank] for rank in ordering])
        preds = [cand_preds[i][0] for i in range(batchsize)]
        return Output(preds, cand_preds)

    def _get_custom_candidates(self, obs):
        """Sets custom candidates for the observation."""
        if not obs.get('eval_labels'):
            # only use custom candidates during training
            cands = []
            cache_sz = len(self.ys_cache) - 1
            if cache_sz < 1:
                return cands
            k = self.opt['neg_samples']
            for i in range(1, k * 3):
                index = random.randint(0, cache_sz)
                cand = self.ys_cache[index]
                if not self.same(obs.get('labels'), cand):
                    cands.append(cand)
                    if len(cands) >= k:
                        break
            if self.opt['parrot_neg']:
                query = self.history.split('\n')[-1]
                cands.append(query)
                return cands
        # if we aren't training, return label candidates if available
        return obs.get('label_candidates')

    def _init_embeddings(self, log=True):
        """Copy embeddings from the pretrained embeddings to the lookuptable.

        :param weight:   weights of lookup table (nn.Embedding/nn.EmbeddingBag)
        :param emb_type: pretrained embedding type
        """
        weight = self.model.lt.weight
        emb_type = self.opt.get('embedding_type', 'random')
        if emb_type == 'random':
            return
        embs, name = self._get_embtype(self, emb_type)
        cnt = 0
        for w, i in self.dict.tok2ind.items():
            if w in embs.stoi:
                vec = self._project_vec(self,
                                        embs.vectors[embs.stoi[w]],
                                        weight.size(1))
                weight.data[i] = vec
                cnt += 1
        if log:
            print('Initialized embeddings for {} tokens ({}%) from {}.'
                  ''.format(cnt, round(cnt * 100 / len(self.dict), 1), name))

    def optimizer_reset(self):
        lr = self.opt['learningrate']
        optim_class = StarspaceTorchAgent.OPTIM_OPTS[self.opt['optimizer']]
        kwargs = {'lr': lr}
        self.optimizer = optim_class(self.model.parameters(), **kwargs)

    def reset(self):
        """Reset observation and episode_done."""
        # self.observation = None
        # self.episode_done = True
        self.observation = {}
        self.history.clear()
        self.replies.clear()
        self.reset_metrics()
        # reset optimizer
        self.optimizer_reset()

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['dict'] = self.dict
        shared['model'] = self.model
        return shared

    def same(self, labels, cand):
        if labels:
            for label in labels:
                if label == cand:
                    return True
        return False

    def input_dropout(self, xs, ys, negs):
        def dropout(x, rate):
            xd = []
            for i in x[0]:
                if random.uniform(0, 1) > rate:
                    xd.append(i)
            if len(xd) == 0:
                # pick one random thing to put in xd
                xd.append(x[0][random.randint(0, x.size(1)-1)])
            return torch.LongTensor(xd).unsqueeze(0)
        rate = self.opt.get('input_dropout')
        xs2 = dropout(xs, rate)
        ys2 = dropout(ys, rate)
        negs2 = []
        for n in negs:
            negs2.append(dropout(n, rate))
        return xs2, ys2, negs2

    def observe(self, observation):
        labels = observation.get('labels')
        if not labels:
            labels = observation.get('eval_labels')
        if labels:
            for label in labels:
                self.add_to_ys_cache(label)
        return super(StarspaceTorchAgent, self).observe(observation)

    def add_to_ys_cache(self, ys):
        if not ys:
            return
        if len(self.ys_cache) < self.ys_cache_sz:
            self.ys_cache.append(ys)
        else:
            ind = random.randint(0, self.ys_cache_sz - 1)
            self.ys_cache[ind] = ys

    def shutdown(self):
        super().shutdown()

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path
        if path and hasattr(self, 'model'):
            data = {}
            data['model'] = self.model.state_dict()
            data['optimizer'] = self.optimizer.state_dict()
            data['opt'] = self.opt
            with open(path, 'wb') as handle:
                torch.save(data, handle)
            with open(path + '.opt', 'w') as handle:
                json.dump(self.opt, handle)

    def load(self, path):
        """Return opt and model states."""
        print('Loading existing model params from ' + path)
        data = torch.load(path, map_location=lambda cpu, _: cpu)
        self.model.load_state_dict(data['model'])
        self.reset()
        self.optimizer.load_state_dict(data['optimizer'])
