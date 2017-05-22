# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Simple example of using a character-level LSTM to train on ParlAI tasks."""

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch

import copy
import os
import random
import time


class Seq2SeqAgent(Agent):
    """Simple agent which uses an LSTM to process incoming text observations."""

    @staticmethod
    def add_cmdline_args(argparser):
        argparser.add_arg('-hs', '--s2s-hiddensize', type=int, default=64,
            help='size of the hidden layers and embeddings')
        argparser.add_arg('-nl', '--s2s-numlayers', type=int, default=2,
            help='number of hidden layers')
        argparser.add_arg('-lr', '--s2s-learningrate', type=float, default=0.5,
            help='learning rate')
        argparser.add_arg('--cuda', action='store_true', default=False,
            help='enable GPUs if available')
        argparser.add_arg('--gpu', type=int, default=-1,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared and 'dictionary' in shared:
            # store dictionary if available
            self.dict = shared['dictionary']
            self.EOS = self.dict.eos_token
            self.EOS_TENSOR = torch.LongTensor(self.dict.parse(self.EOS))

        self.id = 'Seq2Seq'
        hsz = opt['s2s_hiddensize']
        self.hidden_size = hsz
        self.num_layers = opt['s2s_numlayers']
        self.learning_rate = opt['s2s_learningrate']
        self.use_cuda = opt['cuda']
        self.episode_done = True
        self.longest_label = 1

        self.criterion = nn.NLLLoss()
        self.lt = nn.Embedding(len(self.dict), hsz, padding_idx=0,
                               scale_grad_by_freq=True)
        self.encoder = nn.GRU(hsz, hsz, opt['s2s_numlayers'])
        self.decoder = nn.GRU(hsz, hsz, opt['s2s_numlayers'])
        self.d2o = nn.Linear(hsz, len(self.dict))
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

        lr = opt['s2s_learningrate']
        self.emb_optim = optim.Adam(self.lt.parameters(), lr=lr)
        self.enc_optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.dec_optim = optim.Adam(self.decoder.parameters(), lr=lr)
        self.d2o_optim = optim.Adam(self.d2o.parameters(), lr=lr)

        if self.use_cuda:
            self.cuda()

    def parse(self, text):
        return torch.LongTensor(self.dict.txt2vec(text))

    def v2t(self, vec):
        return self.dict.vec2txt(vec)

    def cuda(self):
        self.criterion.cuda()
        self.lt.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.d2o.cuda()
        self.dropout.cuda()
        self.softmax.cuda()

    def hidden_to_idx(self, hidden, drop=False):
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        scores = self.d2o(hidden)
        if drop:
            scores = self.dropout(scores)
        scores = self.softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores

    def zero_grad(self):
        self.emb_optim.zero_grad()
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        self.d2o_optim.zero_grad()

    def update_params(self):
        self.emb_optim.step()
        self.enc_optim.step()
        self.dec_optim.step()
        self.d2o_optim.step()

    def init_zeros(self, bsz=1):
        t = torch.zeros(self.num_layers, bsz, self.hidden_size)
        if self.use_cuda:
            t = t.cuda(async=True)
        return Variable(t)

    def init_ones(self, bsz=1):
        t = torch.zeros(self.num_layers, bsz, self.hidden_size)
        if self.use_cuda:
            t = t.cuda(async=True)
        return Variable(t)

    def init_rand(self, bsz=1):
        t = torch.FloatTensor(self.num_layers, bsz, self.hidden_size)
        t.uniform_(0.05)
        if self.use_cuda:
            t = t.cuda(async=True)
        return Variable(t)

    def observe(self, observation):
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def update(self, xs, ys):
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).t()
        h0 = self.init_zeros(batchsize)
        # c0 = self.init_zeros(batchsize)
        _output, hn = self.encoder(xes, h0)

        # start with EOS tensor for all
        x = self.EOS_TENSOR
        if self.use_cuda:
            x = x.cuda(async=True)
        x = Variable(x)
        xe = self.lt(x).unsqueeze(1)
        xes = torch.cat([xe for _ in range(batchsize)], 1)

        # cn = self.init_zeros(batchsize)
        output_lines = [[] for _ in range(batchsize)]

        self.zero_grad()
        # update model
        loss = 0
        self.longest_label = max(self.longest_label, ys.size(1))
        for i in range(ys.size(1)):
            output, hn = self.decoder(xes, hn)
            preds, scores = self.hidden_to_idx(output, drop=True)
            y = ys.select(1, i)
            loss += self.criterion(scores, y)
            # use the true token as the next input
            xes = self.lt(y).unsqueeze(0)
            # hn = self.dropout(hn)
            for j in range(preds.size(0)):
                token = self.v2t([preds.data[j][0]])
                output_lines[j].append(token)

        loss.backward()
        self.update_params()

        if random.random() < 0.1:
            true = self.v2t(ys.data[0])
            print('loss:', round(loss.data[0], 2), ' '.join(output_lines[0]), '(true: {})'.format(true))
        return output_lines

    def predict(self, xs):
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).t()
        h0 = self.init_zeros(batchsize)
        # c0 = self.init_zeros(batchsize)
        _output, hn = self.encoder(xes, h0)

        # start with EOS tensor for all
        x = self.EOS_TENSOR
        if self.use_cuda:
            x = x.cuda(async=True)
        x = Variable(x)
        xe = self.lt(x).unsqueeze(1)
        xes = torch.cat([xe for _ in range(batchsize)], 1)

        # cn = self.init_zeros(batchsize)
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0
        output_lines = [[] for _ in range(batchsize)]

        while(total_done < batchsize) and max_len < self.longest_label:
            output, hn = self.decoder(xes, hn)
            preds, scores = self.hidden_to_idx(output, drop=False)
            xes = self.lt(preds.t())
            max_len += 1
            for i in range(preds.size(0)):
                if not done[i]:
                    token = self.v2t(preds.data[i])
                    if token == self.EOS:
                        done[i] = True
                        total_done += 1
                    else:
                        output_lines[i].append(token)
        if random.random() < 0.1:
            print('prediction:', ' '.join(output_lines[0]))
        return output_lines

    def batchify(self, obs):
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        batchsize = len(exs)
        parsed = [self.parse(ex['text']) for ex in exs]
        max_x_len = max([len(x) for x in parsed])
        xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
        for i, x in enumerate(parsed):
            offset = max_x_len - len(x)
            for j, idx in enumerate(x):
                xs[i][j + offset] = idx
        if self.use_cuda:
            xs = xs.cuda(async=True)
        xs = Variable(xs)

        ys = None
        if 'labels' in exs[0]:
            labels = [random.choice(ex['labels']) + ' ' + self.EOS for ex in exs]
            parsed = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed)
            ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                ys = ys.cuda(async=True)
            ys = Variable(ys)

        return xs, ys, valid_inds

    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        xs, ys, valid_inds = self.batchify(observations)

        if len(xs) == 0:
            return batch_reply

        # Either train or predict
        if ys is not None:
            predictions = self.update(xs, ys)
        else:
            predictions = self.predict(xs)

        for i in range(len(predictions)):
            batch_reply[valid_inds[i]]['text'] = ' '.join(
                c for c in predictions[i] if c != self.EOS)

        return batch_reply

    def act(self):
        return self.batch_act([self.observation])[0]

    def share(self):
        shared = super().share()
        shared['dictionary'] = self.dict
        return shared


def main():
    # Get command line arguments
    parser = ParlaiParser()
    DictionaryAgent.add_cmdline_args(parser)
    Seq2SeqAgent.add_cmdline_args(parser)
    parser.add_argument('--dict-maxexs', default=100000, type=int)
    opt = parser.parse_args()

    opt['cuda'] = opt['cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        print('[ Using CUDA ]')
        torch.cuda.set_device(opt['gpu'])

    # set up dictionary
    print('Setting up dictionary.')
    dict_tmp_fn = '/tmp/dict_{}.txt'.format(opt['task'])
    if os.path.isfile(dict_tmp_fn):
        opt['dict_loadpath'] = dict_tmp_fn
    dictionary = DictionaryAgent(opt)
    ordered_opt = copy.deepcopy(opt)
    cnt = 0

    if not opt.get('dict_loadpath'):
        for datatype in ['train:ordered', 'valid']:
            # we use train and valid sets to build dictionary
            ordered_opt['datatype'] = datatype
            ordered_opt['numthreads'] = 1
            ordered_opt['batchsize'] = 1
            world_dict = create_task(ordered_opt, dictionary)

            # pass examples to dictionary
            for _ in world_dict:
                cnt += 1
                if cnt > opt['dict_maxexs'] and opt['dict_maxexs'] > 0:
                    print('Processed {} exs, moving on.'.format(
                          opt['dict_maxexs']))
                    # don't wait too long...
                    break
                world_dict.parley()
        dictionary.save(dict_tmp_fn, sort=True)

    agent = Seq2SeqAgent(opt, {'dictionary': dictionary})

    opt['datatype'] = 'train'
    world_train = create_task(opt, agent)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    start = time.time()
    # train / valid loop
    while True:
        print('[ training ]')
        for _ in range(200):  # train for a bit
            world_train.parley()

        print('[ training summary. ]')
        print(world_train.report())

        print('[ validating ]')
        world_valid.reset()
        for _ in world_valid:  # check valid accuracy
            world_valid.parley()

        print('[ validation summary. ]')
        report_valid = world_valid.report()
        print(report_valid)
        if report_valid['accuracy'] > 0.95:
            break

    print('finished in {} s'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    main()
