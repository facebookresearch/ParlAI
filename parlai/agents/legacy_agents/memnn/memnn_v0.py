#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.agents.legacy_agents.seq2seq.seq2seq_v0 import Agent
from parlai.agents.legacy_agents.seq2seq.dict_v0 import DictionaryAgent
from parlai.agents.legacy_agents.seq2seq.utils_v0 import maintain_dialog_history

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

import os
import random

from .modules_v0 import MemNN, Decoder


class MemnnAgent(Agent):
    """
    Memory Network agent.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        arg_group = argparser.add_argument_group('MemNN Arguments')
        arg_group.add_argument(
            '--init-model',
            type=str,
            default=None,
            help='load dict/features/weights/opts from this file',
        )
        arg_group.add_argument(
            '-lr', '--learning-rate', type=float, default=0.01, help='learning rate'
        )
        arg_group.add_argument(
            '--embedding-size', type=int, default=128, help='size of token embeddings'
        )
        arg_group.add_argument(
            '--hops', type=int, default=3, help='number of memory hops'
        )
        arg_group.add_argument(
            '--mem-size', type=int, default=100, help='size of memory'
        )
        arg_group.add_argument(
            '--time-features',
            type='bool',
            default=True,
            help='use time features for memory embeddings',
        )
        arg_group.add_argument(
            '--position-encoding',
            type='bool',
            default=False,
            help='use position encoding instead of bag of words embedding',
        )
        arg_group.add_argument(
            '--output', type=str, default='rank', help='type of output (rank|generate)'
        )
        arg_group.add_argument(
            '--rnn-layers',
            type=int,
            default=2,
            help='number of hidden layers in RNN decoder for generative output',
        )
        arg_group.add_argument(
            '--dropout',
            type=float,
            default=0.1,
            help='dropout probability for RNN decoder training',
        )
        arg_group.add_argument(
            '--optimizer', default='adam', help='optimizer type (sgd|adam)'
        )
        arg_group.add_argument(
            '--no-cuda',
            action='store_true',
            default=False,
            help='disable GPUs even if available',
        )
        arg_group.add_argument(
            '--gpu', type=int, default=-1, help='which GPU device to use'
        )
        arg_group.add_argument(
            '-histr',
            '--history-replies',
            default='label_else_model',
            type=str,
            choices=['none', 'model', 'label', 'label_else_model'],
            help='Keep replies in the history, or not.',
        )
        DictionaryAgent.add_cmdline_args(argparser)
        return arg_group

    def __init__(self, opt, shared=None):
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.device(opt['gpu'])

        if not shared:
            self.id = 'MemNN'
            self.dict = DictionaryAgent(opt)
            self.answers = [None] * opt['batchsize']
            self.model = MemNN(opt, len(self.dict), use_cuda=self.use_cuda)

            self.decoder = None
            if opt['output'] == 'generate' or opt['output'] == 'g':
                self.decoder = Decoder(
                    opt['embedding_size'],
                    opt['embedding_size'],
                    opt['rnn_layers'],
                    opt,
                    self.dict,
                )
            elif opt['output'] != 'rank' and opt['output'] != 'r':
                raise NotImplementedError('Output type not supported.')

            if self.use_cuda and self.decoder is not None:
                # don't call cuda on self.model, it is split cuda and cpu
                self.decoder.cuda()
            if opt['numthreads'] > 1:
                self.model.share_memory()
                if self.decoder is not None:
                    self.decoder.share_memory()
        else:
            self.dict = shared['dict']
            self.model = shared['model']
            self.decoder = shared['decoder']
            if 'threadindex' in shared:
                torch.set_num_threads(1)
                self.answers = [None] * opt['batchsize']
            else:
                self.answers = shared['answers']

        self.opt = opt
        self.mem_size = opt['mem_size']

        self.longest_label = 1
        self.NULL = self.dict.null_token
        self.NULL_IDX = self.dict[self.NULL]
        self.END = self.dict.end_token
        self.END_TENSOR = torch.LongTensor([self.dict[self.END]])
        self.START = self.dict.start_token
        self.START_TENSOR = torch.LongTensor([self.dict[self.START]])

        self.rank_loss = CrossEntropyLoss()
        self.gen_loss = CrossEntropyLoss(ignore_index=self.NULL_IDX)
        if self.use_cuda:
            self.rank_loss.cuda()
            self.gen_loss.cuda()

        if 'train' in self.opt.get('datatype', ''):
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            lr = opt['learning_rate']
            if opt['optimizer'] == 'sgd':
                self.optimizers = {'memnn': optim.SGD(optim_params, lr=lr)}
                if self.decoder is not None:
                    self.optimizers['decoder'] = optim.SGD(
                        self.decoder.parameters(), lr=lr
                    )
            elif opt['optimizer'] == 'adam':
                self.optimizers = {'memnn': optim.Adam(optim_params, lr=lr)}
                if self.decoder is not None:
                    self.optimizers['decoder'] = optim.Adam(
                        self.decoder.parameters(), lr=lr
                    )
            else:
                raise NotImplementedError('Optimizer not supported.')

        if not shared:
            # load model
            # check first for 'init_model' for loading model from file
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                init_model = opt['init_model']
            # next check for 'model_file'
            elif opt.get('model_file') and os.path.isfile(opt['model_file']):
                init_model = opt['model_file']
            else:
                init_model = None
            if init_model is not None:
                print('Loading existing model parameters from ' + init_model)
                self.load(init_model)

        self.history = {}
        self.batch_idx = shared and shared.get('batchindex') or 0
        self.episode_done = True
        self.last_cands, self.last_cands_list = None, None
        super().__init__(opt, shared)

    def share(self):
        shared = super().share()
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['model'] = self.model
        shared['decoder'] = self.decoder
        return shared

    def observe(self, observation):
        """
        Save observation for act.

        If multiple observations are from the same episode, concatenate them.
        """
        self.episode_done = observation['episode_done']
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()

        obs['text'] = maintain_dialog_history(
            self.history,
            obs,
            reply=self.answers[self.batch_idx],
            historyLength=self.opt['mem_size'] + 1,
            useReplies=self.opt['history_replies'],
            dict=self.dict,
            useStartEndIndices=False,
            splitSentences=True,
        )

        self.observation = obs
        self.answers[self.batch_idx] = None
        return obs

    def reset(self):
        self.observation = None
        self.history.clear()
        for i in range(len(self.answers)):
            self.answers[i] = None

    def predict(self, xs, cands, ys=None):
        is_training = ys is not None
        if is_training:
            # Subsample to reduce training time
            cands = [
                list(set(random.sample(c, min(32, len(c))) + self.labels))
                for c in cands
            ]
        else:
            # rank all cands to increase accuracy
            cands = [list(set(c)) for c in cands]

        self.model.train(mode=is_training)
        # Organize inputs for network (see contents of xs and ys in batchify method)
        output_embeddings = self.model(*xs)

        if self.decoder is None:
            scores = self.score(cands, output_embeddings)
            if is_training:
                label_inds = [
                    cand_list.index(self.labels[i]) for i, cand_list in enumerate(cands)
                ]
                if self.use_cuda:
                    label_inds = torch.cuda.LongTensor(label_inds)
                else:
                    label_inds = torch.LongTensor(label_inds)
                loss = self.rank_loss(scores, label_inds)
            predictions = self.ranked_predictions(cands, scores)
        else:
            self.decoder.train(mode=is_training)

            output_lines, loss = self.decode(output_embeddings, ys)
            predictions = self.generated_predictions(output_lines)

        if is_training:
            for o in self.optimizers.values():
                o.zero_grad()
            loss.backward()
            for o in self.optimizers.values():
                o.step()
        return predictions

    def score(self, cands, output_embeddings):
        last_cand = None
        max_len = max([len(c) for c in cands])
        scores = output_embeddings.data.new(len(cands), max_len)
        for i, cand_list in enumerate(cands):
            if last_cand != cand_list:
                candidate_lengths, candidate_indices = to_tensors(cand_list, self.dict)
                candidate_embeddings = self.model.answer_embedder(
                    candidate_lengths, candidate_indices
                )
                if self.use_cuda:
                    candidate_embeddings = candidate_embeddings.cuda()
                last_cand = cand_list
            scores[i, : len(cand_list)] = self.model.score.one_to_many(
                output_embeddings[i].unsqueeze(0), candidate_embeddings
            ).squeeze(0)
        return scores

    def ranked_predictions(self, cands, scores):
        # return [' '] * len(self.answers)
        _, inds = scores.sort(descending=True, dim=1)
        return [
            [cands[i][j] for j in r if j < len(cands[i])] for i, r in enumerate(inds)
        ]

    def decode(self, output_embeddings, ys=None):
        batchsize = output_embeddings.size(0)
        hn = output_embeddings.unsqueeze(0).expand(
            self.opt['rnn_layers'], batchsize, output_embeddings.size(1)
        )
        x = self.model.answer_embedder(torch.LongTensor([1]), self.START_TENSOR)
        xes = x.unsqueeze(1).expand(x.size(0), batchsize, x.size(1))

        loss = 0
        output_lines = [[] for _ in range(batchsize)]
        done = [False for _ in range(batchsize)]
        total_done = 0
        idx = 0
        while (total_done < batchsize) and idx < self.longest_label:
            # keep producing tokens until we hit END or max length for each ex
            if self.use_cuda:
                xes = xes.cuda()
                hn = hn.contiguous()
            preds, scores = self.decoder(xes, hn)
            if ys is not None:
                y = ys[0][:, idx]
                temp_y = y.cuda() if self.use_cuda else y
                loss += self.gen_loss(scores, temp_y)
            else:
                y = preds
            # use the true token as the next input for better training
            xes = self.model.answer_embedder(
                torch.LongTensor(preds.numel()).fill_(1), y
            ).unsqueeze(0)

            for b in range(batchsize):
                if not done[b]:
                    token = self.dict.vec2txt(preds[b])
                    if token == self.END:
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)
            idx += 1
        return output_lines, loss

    def generated_predictions(self, output_lines):
        return [
            [' '.join(c for c in o if c != self.END and c != self.dict.null_token)]
            for o in output_lines
        ]

    def parse(self, memory):
        """Returns:
            query = tensor (vector) of token indices for query
            query_length = length of query
            memory = tensor (matrix) where each row contains token indices for a memory
            memory_lengths = tensor (vector) with lengths of each memory
        """
        query = memory.pop()
        query = torch.LongTensor(query)
        query_length = torch.LongTensor([len(query)])

        if len(memory) == 0:
            memory.append([self.NULL_IDX])

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
        exs = [ex for ex in obs if 'text' in ex and len(ex['text']) > 0]
        valid_inds = [
            i for i, ex in enumerate(obs) if 'text' in ex and len(ex['text']) > 0
        ]
        if not exs:
            return [None] * 4

        parsed = [self.parse(ex['text']) for ex in exs]
        queries = torch.cat([x[0] for x in parsed])
        memories = torch.cat([x[1] for x in parsed])
        query_lengths = torch.cat([x[2] for x in parsed])
        memory_lengths = torch.LongTensor(len(exs), self.mem_size).zero_()
        for i in range(len(exs)):
            if len(parsed[i][3]) > 0:
                memory_lengths[i, -len(parsed[i][3]) :] = parsed[i][3]
        xs = [memories, queries, memory_lengths, query_lengths]

        ys = None
        self.labels = [random.choice(ex['labels']) for ex in exs if 'labels' in ex]
        if len(self.labels) == len(exs):
            parsed = [self.dict.txt2vec(l) for l in self.labels]
            parsed = [torch.LongTensor(p) for p in parsed]
            label_lengths = torch.LongTensor([len(p) for p in parsed]).unsqueeze(1)
            self.longest_label = max(self.longest_label, label_lengths.max())
            padded = [
                torch.cat(
                    (
                        p,
                        torch.LongTensor(self.longest_label - len(p)).fill_(
                            self.END_TENSOR[0]
                        ),
                    )
                )
                for p in parsed
            ]
            labels = torch.stack(padded)
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
        predictions = self.predict(xs, cands, ys)

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
            checkpoint = {}
            checkpoint['memnn'] = self.model.state_dict()
            checkpoint['memnn_optim'] = self.optimizers['memnn'].state_dict()
            if self.decoder is not None:
                checkpoint['decoder'] = self.decoder.state_dict()
                checkpoint['decoder_optim'] = self.optimizers['decoder'].state_dict()
                checkpoint['longest_label'] = self.longest_label
            with open(path, 'wb') as write:
                torch.save(checkpoint, write)

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda cpu, _: cpu)
        self.model.load_state_dict(checkpoint['memnn'])
        self.optimizers['memnn'].load_state_dict(checkpoint['memnn_optim'])
        if self.decoder is not None:
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.optimizers['decoder'].load_state_dict(checkpoint['decoder_optim'])
            self.longest_label = checkpoint['longest_label']


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
    dict_list = list(dict.tok2ind.keys())[1:]  # skip NULL
    cands = []
    for ex in exs:
        if 'label_candidates' in ex:
            cands.append(ex['label_candidates'])
        else:
            cands.append(dict_list)
            if 'labels' in ex:
                cands[-1] += [l for l in ex['labels'] if l not in dict.tok2ind]
    return cands
