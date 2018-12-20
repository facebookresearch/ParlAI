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
        # agent.add_argument('-hist', '--history-length', default=10000, type=int,
        #                    help='Number of past tokens to remember. ')
        # agent.add_argument('-histr', '--history-replies',
        #                    default='label_else_model', type=str,
        #                    choices=['none', 'model', 'label',
        #                             'label_else_model'],
        #                    help='Keep replies in the history, or not.')
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
        # self.fixedCands = False
        # self.fixedX = None
        # if self.opt.get('fixed_candidates_file'):
        #     self.fixedCands_txt = load_cands(self.opt.get('fixed_candidates_file'))
        #     fcs = []
        #     for c in self.fixedCands_txt:
        #         fcs.append(torch.LongTensor(self.parse(c)).unsqueeze(0))
        #     self.fixedCands = fcs
        #     print("[loaded candidates]")

    ##### TO IMPLEMENT
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

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.opt['candidates'], mode='train')

        if self.opt.get('input_dropout', 0) > 0:
            # TODO: fix this input dropout function!!!!!!
            batch, cand_vecs = self.input_dropout(
                batch.text_vec,
                batch.label_vec,
                cand_vecs
            )

        # scores = self.score_candidates(batch, cand_vecs)
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
        import pdb; pdb.set_trace()



    #################

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

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    # def t2v(self, text):
    #     p = self.dict.txt2vec(text)
    #     return torch.LongTensor(p).unsqueeze(1)
    #
    # def v2t(self, vec):
    #     """Convert token indices to string of tokens."""
    #     new_vec = []
    #     for i in vec:
    #         new_vec.append(i)
    #     return self.dict.vec2txt(new_vec)

    # def observe(self, observation):
    #     self.episode_done = observation['episode_done']
    #     # shallow copy observation (deep copy can be expensive)
    #     obs = observation.copy()
    #     obs['text2vec'] = maintain_dialog_history(
    #         self.history, obs,
    #         historyLength=self.opt['history_length'],
    #         useReplies=self.opt['history_replies'],
    #         dict=self.dict, useStartEndIndices=False)
    #     self.observation = obs
    #     return obs

    def same(self, labels, cand):
        for label in labels:
            if label == cand:
                return True
        return False

    def dict_neighbors(self, word, useRHS=False):
        input = self.t2v(word)
        W = self.model.encoder.lt.weight
        q = W[input.data[0][0]]
        if useRHS:
            W = self.model.encoder2.lt.weight
        score = torch.Tensor(W.size(0))
        for i in range(W.size(0)):
            score[i] = torch.nn.functional.cosine_similarity(q, W[i], dim=0).data[0]
        val, ind = score.sort(descending=True)
        for i in range(20):
            print(
                str(ind[i]) + " [" + str(val[i]) + "]: " +
                self.v2t(torch.Tensor([ind[i]]))
            )

    def compute_metrics(self, loss, scores):
        metrics = {}
        pos = scores[0]
        cnt = 0
        for i in range(1, len(scores)):
            if scores[i] >= pos:
                cnt += 1
        metrics['mean_rank'] = cnt
        metrics['loss'] = loss
        return metrics

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

    def predict(self, xs, ys=None, cands=None, cands_txt=None, obs=None):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        is_training = ys is not None
        if is_training:
            negs = self.get_negs(xs, ys)
            if is_training and len(negs) > 0:
                self.model.train()
                self.optimizer.zero_grad()
                if self.opt.get('input_dropout', 0) > 0:
                    xs, ys, negs = self.input_dropout(xs, ys, negs)
                xe, ye = self.model(xs, ys, negs)
                y = -torch.ones(xe.size(0))
                y[0] = 1
                loss = self.criterion(xe, ye, y)
                loss.backward()
                self.optimizer.step()
                pred = nn.CosineSimilarity().forward(xe, ye)
                metrics = self.compute_metrics(loss.item(), pred.data.squeeze())
                return [{'metrics': metrics}]
        else:
            self.model.eval()
            if cands is None or cands[0] is None:
                # cannot predict without candidates.
                if self.fixedCands:
                    cands = [self.fixedCands]
                    cands_txt = [self.fixedCands_txt]
                else:
                    return [{'text': 'I dunno.'}]
                # test set prediction uses fixed candidates
                if self.fixedX is None:
                    xe, ye = self.model(xs, ys, self.fixedCands)
                    self.fixedX = ye
                else:
                    # fixed candidate embed vectors are cached, dont't recompute
                    blah = torch.LongTensor([1])
                    xe, ye = self.model(xs, ys, [blah])
                    ye = self.fixedX
            else:
                # test set prediction uses candidates
                xe, ye = self.model(xs, ys, cands[0])
            pred = nn.CosineSimilarity().forward(xe, ye)
            # This is somewhat costly which we could avoid if we do not evalute ranking.
            # i.e. by only doing: val,ind = pred.max(0)
            val, ind = pred.sort(descending=True)
            # predict the highest scoring candidate, and return it.
            ypred = cands_txt[0][ind.data[0]]
            tc = []
            for i in range(min(100, ind.size(0))):
                tc.append(cands_txt[0][ind.data[i]])
            ret = [{'text': ypred, 'text_candidates': tc}]
            return ret
        return [{'id': self.getID()}]

    # def vectorize(self, observations):
    #     """Convert a list of observations into input & target tensors."""
    #     def valid(obs):
    #         # check if this is an example our model should actually process
    #         return 'text2vec' in obs and len(obs['text2vec']) > 0
    #     try:
    #         # valid examples and their indices
    #         valid_inds, exs = zip(*[(i, ex) for i, ex in
    #                                 enumerate(observations) if valid(ex)])
    #     except ValueError:
    #         # zero examples to process in this batch, so zip failed to unpack
    #         return None, None, None, None
    #
    #     # `x` text is already tokenized and truncated
    #     # sort by length so we can use pack_padded
    #     parsed_x = [ex['text2vec'] for ex in exs]
    #     x_lens = [len(x) for x in parsed_x]
    #     ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])
    #
    #     exs = [exs[k] for k in ind_sorted]
    #     valid_inds = [valid_inds[k] for k in ind_sorted]
    #     parsed_x = [parsed_x[k] for k in ind_sorted]
    #
    #     labels_avail = any(['labels' in ex for ex in exs])
    #
    #     max_x_len = max([len(x) for x in parsed_x])
    #     for x in parsed_x:
    #         x += [self.NULL_IDX] * (max_x_len - len(x))
    #     xs = torch.LongTensor(parsed_x)
    #
    #     # set up the target tensors
    #     ys = None
    #     labels = None
    #     if labels_avail:
    #         # randomly select one of the labels to update on, if multiple
    #         labels = [random.choice(ex.get('labels', [''])) for ex in exs]
    #         # parse each label and append END
    #         parsed_y = [deque(maxlen=self.truncate) for _ in labels]
    #         for dq, y in zip(parsed_y, labels):
    #             dq.extendleft(reversed(self.parse(y)))
    #         max_y_len = max(len(y) for y in parsed_y)
    #         for y in parsed_y:
    #             y += [self.NULL_IDX] * (max_y_len - len(y))
    #         ys = torch.LongTensor(parsed_y)
    #
    #     cands = []
    #     cands_txt = []
    #     if ys is None:
    #         # only build candidates in eval mode.
    #         for o in observations:
    #             if o.get('label_candidates', False):
    #                 cs = []
    #                 ct = []
    #                 for c in o['label_candidates']:
    #                     cs.append(torch.LongTensor(self.parse(c)).unsqueeze(0))
    #                     ct.append(c)
    #                 cands.append(cs)
    #                 cands_txt.append(ct)
    #             else:
    #                 cands.append(None)
    #                 cands_txt.append(None)
    #     return xs, ys, cands, cands_txt

    def add_to_ys_cache(self, ys):
        # TODO: fix this function; we are adding text here
        # also, this should be called at the time of observation
        if ys is None or len(ys) == 0:
            return
        if len(self.ys_cache) < self.ys_cache_sz:
            self.ys_cache.append(copy.deepcopy(ys))
        else:
            ind = random.randint(0, self.ys_cache_sz - 1)
            self.ys_cache[ind] = copy.deepcopy(ys)
    #
    # def batch_act(self, observations):
    #     batchsize = len(observations)
    #     # initialize a table of replies with this agent's id
    #     # convert the observations into batches of inputs and targets
    #     # valid_inds tells us the indices of all valid examples
    #     # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
    #     # since the other three elements had no 'text' field
    #     xs, ys, cands, cands_txt = self.vectorize(observations)
    #     batch_reply = self.predict(xs, ys, cands, cands_txt, observations)
    #     while len(batch_reply) < batchsize:
    #         batch_reply.append({'id': self.getID()})
    #     self.add_to_ys_cache(ys)
    #     return batch_reply

    def shutdown(self):
        # """Save the state of the model when shutdown."""
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
