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
from torch.nn import CrossEntropyLoss

import os
import copy
import random

from .modules import MemNN


class MemnnAgent(Agent):
    """ Memory Network agent.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        argparser.add_arg('-lr', '--learning-rate', type=float, default=0.01,
            help='learning rate')
        argparser.add_arg('--embedding-size', type=int, default=128,
            help='size of token embeddings')
        argparser.add_arg('--hops', type=int, default=3,
            help='number of memory hops')
        argparser.add_arg('--mem-size', type=int, default=100,
            help='size of memory')
        argparser.add_arg('--time-features', type='bool', default=True,
            help='use time features for memory embeddings')
        argparser.add_arg('--position-encoding', type='bool', default=False,
            help='use position encoding instead of bag of words embedding')
        argparser.add_arg('--optimizer', default='adam',
            help='optimizer type (sgd|adam)')
        argparser.add_argument('--no-cuda', action='store_true', default=False,
            help='disable GPUs even if available')
        argparser.add_arg('--gpu', type=int, default=-1,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
        if opt['cuda']:
            print('[ Using CUDA ]')
            torch.cuda.device(opt['gpu'])

        if not shared:
            self.opt = opt
            self.id = 'MemNN'
            self.dict = DictionaryAgent(opt)
            freqs = torch.LongTensor(list(self.dict.freqs().values()))

            self.model = MemNN(opt, freqs)
            self.mem_size = opt['mem_size']
            self.loss_fn = CrossEntropyLoss()
            self.answers = [None] * opt['batchsize']

            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            if opt['optimizer'] == 'sgd':
                self.optimizer = optim.SGD(optim_params, lr=opt['learning_rate'])
            elif opt['optimizer'] == 'adam':
                self.optimizer = optim.Adam(optim_params, lr=opt['learning_rate'])
            else:
                raise NotImplementedError('Optimizer not supported.')

            if opt['cuda']:
                self.model.share_memory()

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])
        else:
            self.answers = shared['answers']

        self.episode_done = True
        self.last_cands, self.last_cands_list = None, None
        super().__init__(opt, shared)

    def share(self):
        shared = super().share()
        shared['answers'] = self.answers
        return shared

    def observe(self, observation):
        observation = copy.copy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text'] if self.observation is not None else ''
            batch_idx = self.opt.get('batchindex', 0)
            if self.answers[batch_idx] is not None:
                prev_dialogue += '\n' + self.answers[batch_idx]
                self.answers[batch_idx] = None
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def update(self, xs, ys, cands):
        self.model.train()
        self.optimizer.zero_grad()

        # Organize inputs for network (see contents of xs and ys in batchify method)
        inputs = [xs[0], xs[1], ys[0], xs[2], xs[3], ys[1]]
        inputs = [Variable(x) for x in inputs]
        output_embeddings, answer_embeddings = self.model(*inputs)
        scores = self.score(cands, output_embeddings, answer_embeddings)

        label_inds = [cand_list.index(self.labels[i]) for i, cand_list in enumerate(cands)]
        if self.opt['cuda']:
            label_inds = Variable(torch.cuda.LongTensor(label_inds))
        else:
            label_inds = Variable(torch.LongTensor(label_inds))

        loss = self.loss_fn(scores, label_inds)
        loss.backward()
        self.optimizer.step()
        return self.ranked_predictions(cands, scores)

    def predict(self, xs, cands):
        self.model.eval()

        # Organize inputs for network (see contents of xs in batchify method)
        inputs = [xs[0], xs[1], None, xs[2], xs[3], None]
        inputs = [Variable(x, volatile=True) for x in inputs]
        output_embeddings, _ = self.model(*inputs)

        scores = self.score(cands, output_embeddings)
        return self.ranked_predictions(cands, scores)

    def score(self, cands, output_embeddings, answer_embeddings=None):
        last_cand = None
        max_len = max([len(c) for c in cands])
        scores = Variable(output_embeddings.data.new(len(cands), max_len))
        for i, cand_list in enumerate(cands):
            if last_cand != cand_list:
                candidate_lengths, candidate_indices = to_tensors(cand_list, self.dict)
                candidate_lengths, candidate_indices = Variable(candidate_lengths), Variable(candidate_indices)
                candidate_embeddings = self.model.answer_embedder(candidate_lengths, candidate_indices)
                if self.opt['cuda']:
                    candidate_embeddings = candidate_embeddings.cuda()
                last_cand = cand_list
            scores[i, :len(cand_list)] = self.model.score.one_to_many(output_embeddings[i].unsqueeze(0), candidate_embeddings)
        return scores

    def ranked_predictions(self, cands, scores):
        _, inds = scores.data.sort(descending=True, dim=1)
        return [[cands[i][j] for j in r if j < len(cands[i])]
                    for i, r in enumerate(inds)]

    def parse(self, text):
        """Returns:
            query = tensor (vector) of token indices for query
            query_length = length of query
            memory = tensor (matrix) where each row contains token indices for a memory
            memory_lengths = tensor (vector) with lengths of each memory
        """
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
        """Returns:
            xs = [memories, queries, memory_lengths, query_lengths]
            ys = [labels, label_lengths] (if available, else None)
            cands = list of candidates for each example in batch
            valid_inds = list of indices for examples with valid observations
        """
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]
        if not exs:
            return [None] * 4

        parsed = [self.parse(ex['text']) for ex in exs]
        queries = torch.cat([x[0] for x in parsed])
        memories = torch.cat([x[1] for x in parsed])
        query_lengths = torch.cat([x[2] for x in parsed])
        memory_lengths = torch.LongTensor(len(exs), self.mem_size).zero_()
        for i in range(len(exs)):
            if len(parsed[i][3]) > 0:
                memory_lengths[i, -len(parsed[i][3]):] = parsed[i][3]
        xs = [memories, queries, memory_lengths, query_lengths]

        ys = None
        self.labels = [random.choice(ex['labels']) for ex in exs if 'labels' in ex]
        if len(self.labels) == len(exs):
            parsed = [self.dict.txt2vec(l) for l in self.labels]
            parsed = [torch.LongTensor(p) for p in parsed]
            label_lengths = torch.LongTensor([len(p) for p in parsed]).unsqueeze(1)
            labels = torch.cat(parsed)
            ys = [labels, label_lengths]

        cands = [ex['label_candidates'] for ex in exs if 'label_candidates' in ex]
        # Use words in dict as candidates if no candidates are provided
        if len(cands) < len(exs):
            cands = build_cands(exs, self.dict)
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

        if xs is None or len(xs[1]) == 0:
            return batch_reply

        # Either train or predict
        if ys is not None:
            predictions = self.update(xs, ys, cands)
        else:
            predictions = self.predict(xs, cands)

        for i in range(len(valid_inds)):
            self.answers[valid_inds[i]] = predictions[i][0]
            batch_reply[valid_inds[i]]['text'] = predictions[i][0]
            batch_reply[valid_inds[i]]['text_candidates'] = predictions[i]
        return batch_reply

    def act(self):
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            model_state = self.model.state_dict()
            optim_state = self.optimizer.state_dict()
            with open(path, 'wb') as write:
                torch.save((model_state, optim_state), write)

    def load(self, path):
        with open(path, 'rb') as read:
            (model, optim) = torch.load(read)
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optim)


def to_tensors(sentences, dictionary):
    lengths = []
    indices = []
    for sentence in sentences:
        tokens = dictionary.txt2vec(sentence)
        lengths.append(len(tokens))
        indices.extend(tokens)
    lengths = torch.LongTensor(lengths)
    indices = torch.LongTensor(indices)
    return lengths, indices


def build_cands(exs, dict):
    dict_list = list(dict.tok2ind.keys())
    cands = []
    for ex in exs:
        if 'label_candidates' in ex:
            cands.append(ex['label_candidates'])
        else:
            cands.append(dict_list)
            if 'labels' in ex:
                cands[-1] += [l for l in ex['labels'] if l not in dict.tok2ind]
    return cands
