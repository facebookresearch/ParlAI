# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import round_sigfigs #, maintain_dialog_history
from parlai.core.thread_utils import SharedTable

from .modules import Kvmemnn

import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch import optim
import torch.nn as nn
import time
from collections import deque

import copy
import os
import random
import math
import pickle

def maintain_dialog_history(history, observation, reply='',
                            historyLength=1, useReplies="labels",
                            dict=None, useStartEndIndices=True,
                            usePersonas=True):
    """Keeps track of dialog history, up to a truncation length.
    Either includes replies from the labels, model, or not all using param 'replies'."""

    def parse(txt):
        txt = txt.lower()
        txt = txt.replace("n't", " not")
        if dict is not None:
            vec =  dict.txt2vec(txt)
            if useStartEndIndices:
                parsed_x = deque([dict[dict.start_token]])
                parsed_x.extend(vec)
                parsed_x.append(dict[dict.end_token])
                return parsed_x
            else:
                return vec
        else:
            return [txt]

    if 'dialog' not in history:
        history['dialog'] = deque(maxlen=historyLength)
        history['persona'] = []
        history['episode_done'] = False
        history['labels'] = []
 
    if history['episode_done']:
        history['dialog'].clear()
        history['persona'] = []
        history['labels'] = []
        history['episode_done'] = False

    # we only keep the last one..that works well for IR model, so..
    history['dialog'].clear()

    if useReplies != 'none':
        if len(history['labels']) > 0:
            r = history['labels'][0]
            history['dialog'].extend(parse(r))
        else: #if useReplies == 'model':
            if reply != '':
                history['dialog'].extend(parse(reply))

    if 'text' in observation:
        txts = observation['text'].split('\n')
        for txt in txts:
            if usePersonas and 'persona:' in txt:
                history['persona'].append(Variable(torch.LongTensor(parse(txt)).unsqueeze(0)))
            else:
                utt = parse(txt)
                history['dialog'].extend(utt)
                history['last_utterance'] = utt

    history['episode_done'] = observation['episode_done']
    if 'labels' in observation:
        history['labels'] = observation['labels']
    elif 'eval_labels' in observation:
        history['labels'] = observation['eval_labels']

    return history['dialog'], history['persona']

def load_cands(path):
    """Load global fixed set of candidate labels that the teacher provides
    every example (the true labels for a specific example are also added to
    this set, so that it's possible to get the right answer).
    """
    if path is None:
        return None
    cands = []
    lines_have_ids = False
    cands_are_replies = False
    cnt = 0
    with open(path) as read:
        for line in read:
            line = line.strip().replace('\\n', '\n')
            if len(line) > 0:
                cnt = cnt + 1
                # If lines are numbered we strip them of numbers.
                if cnt == 1 and line[0:2] == '1 ':
                    lines_have_ids = True
                # If tabs then the label_candidates are all the replies.
                if '\t' in line and not cands_are_replies:
                    cands_are_replies = True
                    cands = []
                if lines_have_ids:
                    space_idx = line.find(' ')
                    line = line[space_idx + 1:]
                    if cands_are_replies:
                        sp = line.split('\t')
                        if len(sp) > 1 and sp[1] != '':
                            cands.append(sp[1])
                    else:
                        cands.append(line)
                else:
                    cands.append(line)
    return cands


class KvmemnnAgent(Agent):
    """Simple implementation of the memnn algorithm with 1 hop
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
        KvmemnnAgent.dictionary_class().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Kvmemnn Arguments')
        agent.add_argument('--hops', type=int, default=1,
                           help='num hops')
        agent.add_argument('--twohop_range', type=int, default=100,
                           help='2 hop range constraint')
        agent.add_argument('--lins', type=int, default=0,
                           help='num lins projecting after hops')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-enorm', '--embeddingnorm', type=float, default=10,
                           help='max norm of word embeddings')
        agent.add_argument('-shareEmb', '--share-embeddings', type='bool', default=True,
                           help='whether LHS and RHS share embeddings')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.005,
                           help='learning rate')
        agent.add_argument('-margin', '--margin', type=float, default=0.3,
                           help='margin')
        agent.add_argument('-opt', '--optimizer', default='sgd',
                           choices=KvmemnnAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-tr', '--truncate', type=int, default=-1,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length.')
        agent.add_argument('-k', '--neg-samples', type=int, default=10,
                           help='number k of negative samples per example')
        agent.add_argument('--parrot-neg', type=int, default=0,
                           help='include query as a negative')
        agent.add_argument('--take-next-utt', type='bool', default=False,
                           help='take next utt')
        agent.add_argument('--tfidf', type='bool', default=False,
                           help='Use frequency based normalization for embeddings.')
        agent.add_argument('-cs', '--cache-size', type=int, default=1000,
                           help='size of negative sample cache to draw from')
        agent.add_argument('-hist', '--history-length', default=100, type=int,
                           help='Number of past tokens to remember. ')
        agent.add_argument('-histr', '--history-replies',
                           default='label', type=str,
                           choices=['none', 'model', 'label'],
                           help='Keep replies in the history, or not.')
        agent.add_argument('--interactive-mode',
                           default=False, type='bool',
                           choices=[True, False])

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt
        self.reset_metrics()
        # all instances needs truncate param
        self.id = 'Kvmemnn'
        self.NULL_IDX = 0
        self.start2 = 99
        # set up tensors once
        self.cands = torch.LongTensor(1, 1, 1)
        self.ys_cache = []
        self.ys_cache_sz = opt['cache_size']
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.history = {}
        if shared:
            if 'threadindex' in shared:
                self.threadindex = shared['threadindex']
            else:
                self.threadindex = 1
            print("[ creating Kvmemnn thread " + str(self.threadindex)  + " ]")
            # set up shared properties
            self.dict = shared['dict']
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']
            self.model = shared['model'] #Kvmemnn(opt, len(self.dict))
            if 'fixedX' in shared:
                self.fixedX = shared['fixedX']
                self.fixedCands = shared['fixedCands']
                self.fixedCands_txt = shared['fixedCands_txt']
                self.fixedCands2 = shared['fixedCands2']
                self.fixedCands_txt2 = shared['fixedCands_txt2']
        else:
            print("[ creating KvmemnnAgent ]")
            # this is not a shared instance of this class, so do full init
            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * 1

            if ((opt['dict_file'] is None and opt.get('model_file')) or 
                os.path.isfile(opt['model_file'] + '.dict')):
                # set default dict-file if not set
                opt['dict_file'] = opt['model_file'] + '.dict'
            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.model = Kvmemnn(opt, len(self.dict), self.dict)
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                self.load(opt['model_file'])
            self.model.share_memory()

            self.fixedCands = False
            self.fixedX = None
            path = opt['model_file'] + '.candspair'
            if os.path.isfile(path):
                print("[loading candidates: " + path + "*]")
                fc = load_cands(path)
                fcs = []
                for c in fc:
                    fcs.append(Variable(torch.LongTensor(self.parse(c)).unsqueeze(0)))
                self.fixedCands = fcs
                self.fixedCands_txt = fc
                fc2 = load_cands(path + "2")
                fcs2 = []
                for c2 in fc2:
                    fcs2.append(Variable(torch.LongTensor(self.parse(c2)).unsqueeze(0)))
                self.fixedCands2 = fcs2
                self.fixedCands_txt2 = fc2
                print("[caching..]")
                xsq = Variable(torch.LongTensor([self.parse('nothing')]))
                xe, ye = self.model(xsq, [], None, self.fixedCands)
                self.fixedX = ye
            print("=init done=")

        self.criterion = torch.nn.CosineEmbeddingLoss(margin=opt['margin'], size_average=False)
        # other options:
        # self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # self.criterion = torch.nn.MultiMarginLoss(p=1, margin=0.1)
        self.reset()
        # can be used to look at embeddings:
        # self.dict_neighbors('coffee')
        self.take_next_utt = True
        self.cands_done = []
        if 'interactive_mode' in opt:
            self.interactiveMode = self.opt['interactive_mode']
        else:
            self.interactiveMode = False
        if self.interactiveMode:
            print("[ Interactive mode ]")


    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'decoder', 'lookuptable', 'attention',
                      'attention_length', 'fixed_candidates_file'}
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def parse(self, text):
        """Convert string to token indices."""
        text = text.lower()
        text = text.replace("n't", " not")
        return self.dict.txt2vec(text)

    def t2v(self, text):
        p = self.dict.txt2vec(text)
        return Variable(torch.LongTensor(p).unsqueeze(1))

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        if type(vec) == Variable:
            vec = vec.data
        if type(vec) == torch.LongTensor and vec.dim() == 2:
            vec = vec.squeeze(0)
        new_vec = []
        for i in vec:
            new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def zero_grad(self):
        """Zero out optimizer."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        self.optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True
        # set up optimizer
        lr = self.opt['learningrate']
        optim_class = KvmemnnAgent.OPTIM_OPTS[self.opt['optimizer']]
        kwargs = {'lr': lr}
        self.optimizer = optim_class(self.model.parameters(), **kwargs)


    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['model'] = self.model
        if self.fixedX is not None:
            shared['fixedX'] = self.fixedX
            shared['fixedCands'] = self.fixedCands
            shared['fixedCands_txt'] = self.fixedCands_txt
            shared['fixedCands2'] = self.fixedCands2
            shared['fixedCands_txt2'] = self.fixedCands_txt2
        return shared

    def observe(self, observation):
        self.episode_done = observation['episode_done']
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        obs['query'],obs['mem'] = maintain_dialog_history(
            self.history, obs,
            historyLength=self.opt['history_length'],
            useReplies=self.opt['history_replies'],
            dict=self.dict, useStartEndIndices=False)
        self.observation = obs
        return obs

    def report2(self):
        def clip(f):
            return round_sigfigs(f)

        metrics = self.metrics
        if metrics['total'] == 0:
            report = { 'mean_rank': self.opt['neg_samples'] }
        else:
            maxn = 0
            for i in range(100):
                index =  random.randint(0, self.model.lt.weight.size(0)-1)
                n = self.model.lt.weight[5].norm(2).data[0]
                if n > maxn:
                    maxn = n

            report = { 'total': clip(metrics['total_total']),
                       'loss': clip(metrics['loss'] / metrics['total']),
                       'mean_rank': clip(metrics['mean_rank'] / metrics['total']),
                       'mlp_time': clip(metrics['mlp_time'] / metrics['total']),
                       'tot_time': clip(metrics['tot_time'] / metrics['total']),
                       'max_norm': clip(n),
                       }
        return report

    def reset_metrics(self, keep_total=False):
        if keep_total:
            self.metrics = { 'total':0, 'mean_rank':0, 'loss':0,
                             'total_total':self.metrics['total_total'],
                             'mlp_time':0, 'tot_time':0,
                             'max_weight':0, 'mean_weight':0}
        else:
            self.metrics = { 'total_total':0, 'mean_rank':0, 'total':0,
                             'mlp_time':0, 'tot_time':0, 'loss':0,
                             'max_weight':0, 'mean_weight':0}

    def compute_metrics(self, loss, scores, mlp_time, non_mlp_time):
        metrics = {}
        pos = scores[0]
        cnt = 0
        for i in range(1, len(scores)):
            if scores[i] >= pos:
                cnt += 1
        metrics['mean_rank'] = cnt
        metrics['loss'] = loss
        metrics['tot_time'] = mlp_time + non_mlp_time
        metrics['mlp_time'] = mlp_time
        return metrics

    def same(self, y1, y2):
        if len(y1.squeeze()) != len(y2.squeeze()):
            return False
        if abs((y1.squeeze()-y2.squeeze()).sum().data.sum()) > 0.00001:
            return False
        return True

    def get_negs(self, xs, ys):
        negs = []
        #for neg in self.ys_cache:
        cache_sz = len(self.ys_cache) - 1
        if cache_sz < 1:
            return negs
        k = self.opt['neg_samples']
        for i in range(1, k * 3):
            index =  random.randint(0, cache_sz)
            neg = self.ys_cache[index]
            if not self.same(ys, neg):
                negs.append(neg)
                if len(negs) >= k:
                    break
        if self.opt['parrot_neg'] > 0:
            utt = self.history['last_utterance']
            if len(utt) > 2:
                query = Variable(torch.LongTensor(utt).unsqueeze(0))
                #print(self.v2t(query.squeeze(0)))
                negs.append(query)
        return negs

    def dict_neighbors(self, word, useRHS=False):
        input = self.t2v(word)
        W = self.model.encoder.lt.weight
        q = W[input.data[0][0]]
        if useRHS:
            W = self.model.encoder2.lt.weight
        score = torch.Tensor(W.size(0))
        for i in range(W.size(0)):
            score[i] = torch.nn.functional.cosine_similarity(q, W[i], dim=0).data[0]
        val,ind=score.sort(descending=True)
        for i in range(20):
            print(str(ind[i]) + " [" + str(val[i]) + "]: " + self.v2t(torch.Tensor([ind[i]])))

    def predict(self, xs, ys=None, cands=None, cands_txt=None, obs=None):
        """Produce a prediction from our model.
        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        self.start = time.time()
        is_training = ys is not None
        if is_training: #
            text_cand_inds, loss_dict = None, None
            negs = self.get_negs(xs, ys)
            if is_training and len(negs) > 0: # and self.opt['learningrate'] > 0:
                self.model.train()
                self.zero_grad()
                xe, ye = self.model(xs, obs[0]['mem'], ys, negs)
                y = Variable(-torch.ones(xe.size(0)))
                y[0]= 1
                loss = self.criterion(xe, ye, y)
                loss.backward()
                self.update_params()
                rest = 0
                if self.start2 != 99:
                    rest = self.start-self.start2
                self.start2 = time.time()
                pred = nn.CosineSimilarity().forward(xe,ye)
                metrics = self.compute_metrics(loss.data[0],
                    pred.data.squeeze(), self.start2-self.start, rest)
                return [{'metrics':metrics}]
        else:
            fixed = False
            if hasattr(self, 'fixedCands') and self.fixedCands:
                self.take_next_utt=True
                self.twohoputt=True
                self.tricks=True
            else:
                self.take_next_utt = False
                self.twohoputt=False
                self.tricks=False
            if cands is None or cands[0] is None or self.take_next_utt:
                # cannot predict without candidates.
                if self.fixedCands or self.take_next_utt:
                    cands_txt2 = [self.fixedCands_txt2]
                    fixed = True
                else:
                    return [{}]
            # test set prediction uses candidates
            self.model.eval()
            if fixed:
                if obs[0]['episode_done']:
                    self.cands_done = []

                if xs is None:
                    xs = Variable(torch.LongTensor([self.parse('nothing')]))
                xs = xs.clone()
                if self.tricks:
                    vv=self.history['last_utterance']
                    if len(vv) == 0:
                        xsq = Variable(torch.LongTensor([self.parse('nothing')]))
                    else:
                        xsq = Variable(torch.LongTensor([vv]))

                else:
                    xsq = xs
                mems= obs[0]['mem']
                if self.tricks:
                    mems = []
                if self.fixedX is None:
                    xe, ye = self.model(xsq, mems, ys, self.fixedCands)
                    self.fixedX = ye
                else:
                    # fixed cand embed vectors are cached, dont't recompute
                    blah = Variable(torch.LongTensor([1]))
                    xe, ye = self.model(xsq, mems, ys, [blah])
                    ye = self.fixedX
                pred = nn.CosineSimilarity().forward(xe,ye)
                origpred = pred
                val,ind=pred.sort(descending=True)
                origind = ind
                ypredorig = self.fixedCands_txt[ind.data[0]] # match
                ypred = cands_txt2[0][ind.data[0]] # reply to match
                tc = [ypred]
                if self.twohoputt:
                    # now we rerank original cands against this prediction
                    zq = []
                    z = []
                    ztxt = []
                    newwords = {}
                    if 'twohop-range' not in self.opt:
                        r=100
                    else:
                        r=self.opt['twohop-range']
                    for i in range(r):
                        c = self.fixedCands2[ind.data[i]]
                        ctxt = self.fixedCands_txt2[ind.data[i]]
                        if i < 10:
                            zq.append(c)
                        z.append(c)
                        ztxt.append(ctxt)
                        for w in c[0]:
                            newwords[w.data[0]] = True
                    xs2 = torch.cat(zq, 1)

                if self.interactiveMode and self.twohoputt:
                    # used for nextutt alg in demo mode, get 2nd hop
                    blah = Variable(torch.LongTensor([1]))
                    if self.tricks:
                        xe, ye = self.model(xs2, obs[0]['mem'], ys, z)
                    else:
                        xe, ye = self.model(xs2, obs[0]['mem'], ys, [blah])
                        ye = self.fixedX
                    pred = nn.CosineSimilarity().forward(xe,ye)
                    for c in self.cands_done:
                        for i in range(len(ztxt)):
                            if ztxt[i] == c:
                                # interactive heuristic: don't repeat yourself
                                pred[i] = -1000
                    val,ind=pred.sort(descending=True)
                    # predict the highest scoring candidate, and return it.
                    print("   [query:          " + self.v2t(xsq) + "]")
                    ps = []
                    for c in obs[0]['mem']:
                        ps.append(self.v2t(c))
                    print("   [persona:        " + '|'.join(ps) + "]")
                    print("   [1st hop qmatch: " + ypredorig + "]")
                    print("   [1st hop nextut: " + ypred + "]")
                    if self.tricks:
                        ypred = ztxt[ind.data[0]] # match
                        self.cands_done.append(ypred)
                    else:
                        ypred = self.fixedCands_txt[ind.data[0]] # match
                        ypred2 = cands_txt2[0][ind.data[0]] # reply to match
                        self.cands_done.append(ind.data[0])
                        print("   [2nd hop nextut: " + ypred2 + "]")
                    tc = [ypred]
                    self.history['labels'] = [ypred]
                    ret = [{'text': ypred, 'text_candidates': tc }]
                    return ret
                elif self.take_next_utt and not self.interactiveMode:
                    xe, ye = self.model(xs2, obs[0]['mem'], ys, cands[0])
                    pred = nn.CosineSimilarity().forward(xe, ye)
                    xe, ye = self.model(xs, obs[0]['mem'], ys, cands[0])
                    origpred = nn.CosineSimilarity().forward(xe,ye)
                    if 'alpha' not in self.opt:
                        alpha=0.1
                    else:
                        alpha=self.opt['alpha']
                    pred = alpha*pred + 1*origpred
                    val,ind=pred.sort(descending=True)
                    # predict the highest scoring candidate, and return it.
                    ypred = cands_txt[0][ind.data[0]] # match
                    tc = []
                    for i in range(len(ind)):
                        tc.append(cands_txt[0][ind.data[i]])
            else:
                xe, ye = self.model(xs, obs[0]['mem'], ys, cands[0])
                pred = nn.CosineSimilarity().forward(xe,ye)
                val,ind=pred.sort(descending=True)
                ypred = cands_txt[0][ind.data[0]] # match
                tc = []
                for i in range(min(100, ind.size(0))):
                    tc.append(cands_txt[0][ind.data[i]])
            ret = [{'text': ypred, 'text_candidates': tc }]
            return ret
        return [{}]

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        def valid(obs):
            # check if this is an example our model should actually process
            return 'query' in obs and len(obs['query']) > 0
        try:
            # valid examples and their indices
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None, None

        # set up the input tensors
        bsz = len(exs)

        # `x` text is already tokenized and truncated
        # sort by length so we can use pack_padded
        parsed_x = [ex['query'] for ex in exs]
        x_lens = [len(x) for x in parsed_x]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed_x = [parsed_x[k] for k in ind_sorted]

        labels_avail = any(['labels' in ex for ex in exs])

        max_x_len = max([len(x) for x in parsed_x])
        for x in parsed_x:
            x += [[self.NULL_IDX]] * (max_x_len - len(x))
        xs = torch.LongTensor(parsed_x)
        xs = Variable(xs)

        # set up the target tensors
        ys = None
        labels = None
        if labels_avail:
            # randomly select one of the labels to update on, if multiple
            labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            # parse each label and append END
            parsed_y = [deque(maxlen=self.truncate) for _ in labels]
            for dq, y in zip(parsed_y, labels):
                dq.extendleft(reversed(self.parse(y)))
            max_y_len = max(len(y) for y in parsed_y)
            for y in parsed_y:
                y += [self.NULL_IDX] * (max_y_len - len(y))
            ys = torch.LongTensor(parsed_y)
            ys = Variable(ys)

        cands = []
        cands_txt = []
        if ys is None:
            # only build candidates in eval mode.
            for o in observations:
                if 'label_candidates' in o:
                    cs = []
                    ct = []
                    for c in o['label_candidates']:
                        cs.append(Variable(torch.LongTensor(self.parse(c)).unsqueeze(0)))
                        ct.append(c)
                    cands.append(cs)
                    cands_txt.append(ct)
                else:
                    cands.append(None)
                    cands_txt.append(None)
        return xs, ys, cands, cands_txt

    def add_to_ys_cache(self, ys):
        if ys is None or len(ys) == 0:
            return
        if len(self.ys_cache) < self.ys_cache_sz:
            self.ys_cache.append(copy.deepcopy(ys))
        else:
            ind = random.randint(0, self.ys_cache_sz - 1)
            self.ys_cache[ind] = copy.deepcopy(ys)

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        if batchsize == 0 or 'text' not in observations[0]:
            return [{ 'text': 'dunno' }]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, cands, cands_txt = self.batchify(observations)
        batch_reply = self.predict(xs, ys, cands, cands_txt, observations)
        self.add_to_ys_cache(ys)
        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]


    def shutdown(self):
        #"""Save the state of the model when shutdown."""
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
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Return opt and model states."""
        with open(path, 'rb') as read:
            print('Loading existing model params from ' + path)
            data = torch.load(read)
            self.model.load_state_dict(data['model'])
            self.reset()
            self.optimizer.load_state_dict(data['optimizer'])
            self.opt = self.override_opt(data['opt'])
            
