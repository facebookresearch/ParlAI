# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

import torch
from torch import optim
from torch.autograd import Variable

import copy
import random

from .modules import ForwardPrediction, WarpLossWithBatch, MemNN


class MemnnAgent(Agent):
    # TODO
    """ Memory Network agent.
    """

    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        argparser.add_arg('-lr', '--learning-rate', type=float, default=0.1,
            help='learning rate')
        argparser.add_arg('--embedding-size', type=int, default=256,
            help='size of token embeddings')
        argparser.add_arg('--hops', type=int, default=3,
            help='number of memory hops')
        argparser.add_arg('--mem-size', type=int, default=10,
            help='size of memory')
        argparser.add_arg('--mem-temp', type=float, default=1,
            help='softmax temperature in memory hops')
        argparser.add_arg('--forward-temp', type=float, default=0.25,
            help='softmax temperature in forward hops')
        argparser.add_arg('--forward-time', type=int, default=1,
            help='predict sentences at time t + <forward-time>')
        argparser.add_arg('--gating', type='bool', default=True,
            help='use memory gating at each hop')
        argparser.add_arg('--speaker-features', type='bool', default=True,
            help='use speaker features for memory embeddings')
        argparser.add_arg('--time-features', type='bool', default=True,
            help='use time features for memory embeddings')
        argparser.add_arg('--margin', type=float, default=0.01,
            help='ranking loss margin')
        argparser.add_arg('--optimizer', default='adagrad',
            help='optimizer type (sgd|adagrad)')
        argparser.add_arg('--pretokenized-cands', type='bool', default=False,
            help='candidates are already tokenized (e.g., bAbI)')
        argparser.add_arg('--weighting-scheme', default='smooth',
            help='feature weighting scheme (log|smooth)')
        argparser.add_arg('--score', default='dot',
            help='scoring function (dot|concat|triple)')
        argparser.add_arg('--score-hidden-size', type=int, default=200,
            help='hidden size of score function mlp')
        argparser.add_arg('--triple-beam-size', type=int, default=100,
            help='beam size for triple search')
        argparser.add_arg('--cuda', action='store_true',
            help='use GPUs if available')
        argparser.add_arg('--gpu', type=int, default=-1,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        if opt['cuda']:
            print('[ Using CUDA ]')
            torch.cuda.device(opt['gpu'])

        self.dict = DictionaryAgent(opt)
        freqs = torch.LongTensor(list(self.dict.freqs().values()))

        self.model = MemNN(opt, freqs)
        self.mem_size = opt['mem_size']

        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optim_params, lr=opt['learning_rate'])
        elif opt['optimizer'] == 'adagrad':
            self.optimizer = optim.Adagrad(optim_params, lr=opt['learning_rate'])
        else:
            raise NotImplementedError

        self.model.share_memory()
        self.optimizer.share_memory()

        self.episode_done = True
        self.last_cands, self.last_cands_list = None, None
        self.opt = opt
        super().__init__(opt, shared)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            # TODO include last answer into memory as well?
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def update(self, xs, ys, cands):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = [xs[0], xs[1], ys[0], xs[2], xs[3], ys[1]]
        inputs = [Variable(x) for x in inputs]
        query_embeddings, answer_embeddings = self.model(*inputs)

        loss_fn = build_loss(self.opt, self.model)
        loss = loss_fn(query_embeddings, answer_embeddings)
        loss.backward()
        self.optimizer.step()

        return self.ranked_predictions(cands, query_embeddings)

    def predict(self, xs, cands):
        self.model.eval()

        inputs = [xs[0], xs[1], None, xs[2], xs[3], None]
        inputs = [Variable(x, volatile=True) for x in inputs]
        query_embeddings, _ = self.model(*inputs)

        return self.ranked_predictions(cands, query_embeddings)

    def ranked_predictions(self, cands, embeddings):
        inds = [None] * len(cands)
        last_cand, candidate_lengths, candidate_indices, candidate_embeddings = [None] * 4
        for i, cand in enumerate(cands):
            # Avoid recalculating embeddings if candidate is the same
            if last_cand != cand:
                candidate_lengths, candidate_indices = to_tensors(cand, self.dict, self.opt['pretokenized_cands'])
                candidate_embeddings = self.model.embed('L', candidate_lengths, candidate_indices)
                if self.opt['cuda']:
                    candidate_embeddings = candidate_embeddings.cuda(async=True)
                last_cand = cand
            _, inds[i] = self.model.score.one_to_many(embeddings[i].unsqueeze(0),
                    candidate_embeddings).data.sort(descending=True, dim=1)
            inds[i].squeeze_()
        return [[cands[i][j] for j in r] for i, r in enumerate(inds)]

    def parse(self, text):
        sp = text.split('\n')
        query_sentence = sp[-1]
        query = self.dict.txt2vec(query_sentence)
        query = torch.LongTensor(query)
        query_length = torch.LongTensor([len(query)])

        sp = sp[:-1]
        sentences = []
        for s in sp:
            sentences.extend(s.split('\t'))
        if len(sentences) == 0:
            sentences.append(self.dict.null_token)

        num_mems = min(self.mem_size, len(sentences))
        memory_sentences = sentences[-num_mems:]
        memory = [self.dict.txt2vec(s) for s in memory_sentences]
        memory = [torch.LongTensor(m) for m in memory]
        memory_lengths = torch.LongTensor([len(m) for m in memory])
        memory = torch.cat(memory)

        return (query, memory, query_length, memory_lengths)

    def batchify(self, obs):
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        parsed = [self.parse(ex['text']) for ex in exs]
        queries = torch.cat([x[0] for x in parsed])
        memories = torch.cat([x[1] for x in parsed])
        query_lengths = torch.cat([x[2] for x in parsed])
        memory_lengths = torch.LongTensor(len(exs), self.mem_size).zero_()
        for i in range(len(exs)):
            if len(parsed[i][3]) > 0:
                memory_lengths[i, -len(parsed[i][3]):] = parsed[i][3]
        xs = [memories, queries, memory_lengths, query_lengths]

        # TODO Implement forward time for more than 1 prediction?
        ys = None
        labels = [random.choice(ex['labels']) for ex in exs if 'labels' in ex]
        if labels:
            parsed = [self.dict.txt2vec(l) for l in labels]
            parsed = [torch.LongTensor(p) for p in parsed]
            label_lengths = torch.LongTensor([len(p) for p in parsed]).unsqueeze(1)
            labels = torch.cat(parsed)
            ys = [labels, label_lengths]

        cands = [o['label_candidates'] for o in obs if 'label_candidates' in o]
        # Avoid rebuilding candidate list every batch if its the same
        if self.last_cands != cands:
            self.last_cands = cands
            self.last_cands_list = [list(c) for c in cands]
        cands = self.last_cands_list

        return xs, ys, cands, valid_inds

    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        xs, ys, cands, valid_inds = self.batchify(observations)

        if len(xs[1]) == 0:
            return batch_reply

        # Either train or predict
        if ys is not None:
            predictions = self.update(xs, ys, cands)
        else:
            predictions = self.predict(xs, cands)

        for i in range(len(valid_inds)):
            batch_reply[valid_inds[i]]['text'] = predictions[i][0]
            batch_reply[valid_inds[i]]['text_candidates'] = predictions[i]

        return batch_reply

    def act(self):
        temp = self.batch_act([self.observation])[0]
        return temp

    def save(self, path):
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()

        with open(path, 'wb') as write:
            torch.save((model_state, optim_state), write)

    def load(self, path):
        with open(path, 'rb') as read:
            (model, optim) = torch.load(read)
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optim)


def build_loss(opt, model):
    loss = WarpLossWithBatch(opt['margin'], model.score)

    if hasattr(model, 'forward_hop'):
        forward_prediction = ForwardPrediction(model.forward_hop)

    def loss_fn(query_embeddings, answer_embeddings):
        if opt['score'] == 'triple':
            return loss(query_embeddings, answer_embeddings[:, 0], answer_embeddings[:, 1])
        if hasattr(model, 'forward_hop'):
            query_embeddings = forward_prediction(query_embeddings, answer_embeddings[:, :-1])
        return loss(query_embeddings, answer_embeddings[:, -1])

    return loss_fn


def to_tensors(sentences, dictionary, pretokenized=False):
    lengths = []
    indices = []
    for sentence in sentences:
        if not pretokenized:
            sentence = ' '.join(dictionary.tokenize(sentence))
        tokens = dictionary.txt2vec(sentence)
        lengths.append(len(tokens))
        indices.extend(tokens)
    lengths = torch.LongTensor(lengths)
    indices = torch.LongTensor(indices)
    return lengths, indices
