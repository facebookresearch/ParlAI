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
        agent.add_argument('--utts-retrieved', type=int, default=10,
                           help='number o retrieved utts per example')
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
        agent.add_argument('-fixedCands', '--fixed-candidates-file',
                           default=None, type=str,
                           help='File of cands to use for prediction')

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt
        #opt['learningrate'] = 0.5

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
            self.threadindex = shared['threadindex']
            print("[ creating Kvmemnn thread " + str(self.threadindex)  + " ]")
            # set up shared properties
            self.dict = shared['dict']
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']
            self.model = shared['model']
        else:
            print("[ creating KvmemnnAgent ]")
            # this is not a shared instance of this class, so do full init
            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * 1

            if opt['dict_file'] is None and opt.get('model_file'):
                # set default dict-file if not set
                opt['dict_file'] = opt['model_file'] + '.dict'
            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)

            self.model = Kvmemnn(opt, len(self.dict), self.dict)
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                self.load(opt['model_file'])
            self.model.share_memory()

        # set up modules
        #self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        #self.criterion = torch.nn.MultiMarginLoss(p=1, margin=0.1)
        self.criterion = torch.nn.CosineEmbeddingLoss(margin=opt['margin'], size_average=False)
        self.reset()
        #self.dict_neighbors('coffee')

        #self.opt['fixed-candidates-file'] = "/Users/jase/Desktop/data_20170815_valid_candidates.txt"

        self.take_next_utt = True
        self.opt['fixed_candidates_file'] = "data/personachat/candspair.txt"
        self.fixedCands = False
        self.fixedX = None
        self.cands_done = []
        if self.opt.get('fixed_candidates_file'):
            print("[loading candidates..]")
            fc = load_cands(self.opt.get('fixed_candidates_file'))
            fcs = []
            for c in fc:
                fcs.append(Variable(torch.LongTensor(self.parse(c)).unsqueeze(0)))
            self.fixedCands = fcs
            self.fixedCands_txt = fc
            fc2 = load_cands(self.opt.get('fixed_candidates_file') + "2")
            fcs2 = []
            for c2 in fc2:
                fcs2.append(Variable(torch.LongTensor(self.parse(c2)).unsqueeze(0)))
            self.fixedCands2 = fcs2
            self.fixedCands_txt2 = fc2
            
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

    def load_ir_cache(self):
        if hasattr(self, 'IR_matrix'):
            return
        if False:
            self.IR_matrix = torch.LongTensor(len(self.fixedCands_txt), 100)
            m = self.IR_matrix
            m = m - 1
            for i in range(80):
                print(i)
                path = 'data/personachat/ir_cache_' + self.opt['datatype'] + "_" + str(i) + ".vecs" 
                with open(path, 'rb') as handle:
                    x = torch.load(handle)
                for j in x.keys():
                    m[j] = x[j].data
            path = 'data/personachat/ir_cache_' + self.opt['datatype'] + "_allvecs"
            with open(path, 'wb') as handle:
                torch.save(m, handle)
        else:
            path = 'data/personachat/ir_cache_' + self.opt['datatype'] + "_allvecs"
            with open(path, 'rb') as handle:
                self.IR_matrix = torch.load(handle)
            m = self.IR_matrix    

    def next_utts(self, obs):
        index = int(obs['reward'])
        utts = self.IR_matrix[index]
        z1 = []
        z2 = []
        for i in range(self.opt['utts_retrieved']):
            z1.append(self.fixedCands[utts[i]])
            z2.append(self.fixedCands2[utts[i]])
        return [z1, z2]
            
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
                #import pdb; pdb.set_trace()
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
        self.load_ir_cache()
        self.start = time.time()
        is_training = ys is not None
        if hasattr(self, 'ir_cachez'):
            is_training = False
        if is_training: #
            text_cand_inds, loss_dict = None, None
            negs = self.get_negs(xs, ys)
            if is_training and len(negs) > 0: # and self.opt['learningrate'] > 0:
                self.model.train()
                self.zero_grad()
                xe, ye = self.model(xs, obs[0]['mem'], ys, negs, self.next_utts(obs[0]))
                if False:
                    # print example
                    print("inp: " + self.v2t(xs.squeeze()))
                    print("pos: " + self.v2t(ys.squeeze()))
                    for c in negs:
                        print("neg: " + self.v2t(c.squeeze()))
                    print("---")
                    #import pdb; pdb.set_trace()
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
            self.take_next_utt=False #True
            self.twohoputt=False #True
            self.tricks=False #True
            self.metricEval=True #False
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
                
                xs = xs.clone()
                if self.tricks:
                    if self.metricEval:
                        xsq = cands[0][20]
                    else:
                        vv=self.parse(obs[0]['text'])
                        xsq = Variable(torch.LongTensor([vv]))
                else:
                    xsq = xs
                if self.fixedX is None:
                    xe, ye = self.model(xsq, obs[0]['mem'], ys, self.fixedCands)
                    self.fixedX = ye
                else:
                    # fixed cand embed vectors are cached, dont't recompute
                    blah = Variable(torch.LongTensor([1]))
                    xe, ye = self.model(xsq, obs[0]['mem'], ys, [blah])
                    ye = self.fixedX
                pred = nn.CosineSimilarity().forward(xe,ye)
                origpred = pred
                for i in range(pred.size(0)):
                    v = pred.data[i]
                    if v > 1.5:
                        v = 0
                        #alpha=10
                        #v = v - alpha*(1-v)
                    pred.data[i] = v
                val,ind=pred.sort(descending=True)
                # predict the highest scoring candidate, and return it.
                ypredorig = self.fixedCands_txt[ind.data[0]] # match
                ypred = cands_txt2[0][ind.data[0]] # reply to match
                tc = [ypred]
                if self.twohoputt:
                    # now we rerank original cands against this pred
                    z = []
                    newwords = {}
                    for i in range(10):
                        c = self.fixedCands2[ind.data[i]]
                        z.append(c)
                        for w in c[0]:
                            newwords[w.data[0]] = True
                    #z.append(xs)
                    xs2 = torch.cat(z, 1)

                if self.twohoputt and not self.metricEval:
                    # used for nextutt alg in demo mode, get 2nd hop
                    blah = Variable(torch.LongTensor([1]))
                    xe, ye = self.model(xs2, obs[0]['mem'], ys, [blah])
                    ye = self.fixedX
                    pred = nn.CosineSimilarity().forward(xe,ye)
                    #pred = 0.6*pred + origpred # emphasizes query too much
                    for c in self.cands_done:
                        pred[c] = -1000
                    val,ind=pred.sort(descending=True)
                    # predict the highest scoring candidate, and return it.
                    print("   [query:          " + self.v2t(xs) + "]")
                    print("   [1st hop qmatch: " + ypredorig + "]")
                    print("   [1st hop nextut: " + ypred + "]")
                    ypred = self.fixedCands_txt[ind.data[0]] # match
                    ypred2 = cands_txt2[0][ind.data[0]] # reply to match
                    self.cands_done.append(ind.data[0])
                    print("   [2nd hop nextut: " + ypred2 + "]")
                    tc = [ypred]
                    self.history['labels'] = [ypred]
                    ret = [{'text': ypred, 'text_candidates': tc }]
                    return ret
                elif self.take_next_utt and self.metricEval:
                    xe, ye = self.model(xs2, obs[0]['mem'], ys, cands[0])
                    pred = nn.CosineSimilarity().forward(xe, ye)
                    xe, ye = self.model(xs, obs[0]['mem'], ys, cands[0])
                    origpred = nn.CosineSimilarity().forward(xe,ye)
                    pred = 0.6*pred + 1*origpred # ~0.12 or 0.13
                    pred[20]= 0
                    val,ind=pred.sort(descending=True)
                    # predict the highest scoring candidate, and return it.
                    ypred = cands_txt[0][ind.data[0]] # match
                    tc = []
            else:
                xe, ye = self.model(xs, obs[0]['mem'], ys, cands[0])
                pred = nn.CosineSimilarity().forward(xe,ye)
                if len(pred) == 21:
                    # remove parrot for now..
                    pred[20]=-1000
                val,ind=pred.sort(descending=True)
                # predict the highest scoring candidate, and return it.
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
        #path = self.opt.get('model_file', None)
        #if path is not None:
        #    self.save(path + '.shutdown_state')
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
