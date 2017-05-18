# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Simple example of using a character-level LSTM to train on ParlAI tasks."""

from parlai.core.params import ParlaiParser
from parlai.core.agents import Agent
from parlai.core.worlds import create_task

from torch.autograd import Variable
import torch.nn as nn
import torch

import random
import string
import time


class Seq2SeqAgent(Agent):
    """Simple agent which uses an RNN to process incoming text observations."""

    # preparation for turning letters to indicies
    EOS = '\r'
    UNK = '\v'
    VOCAB = {k: i for i, k in
             enumerate('~' + string.ascii_letters + ' ?!.,;"\'\n' + UNK + EOS)}
    TOKENS = {i: k for k, i in VOCAB.items()}
    VOCAB_LEN = len(VOCAB)
    EOS_TENSOR = Variable(torch.LongTensor([VOCAB[EOS]]))

    @staticmethod
    def add_cmdline_args(argparser):
        argparser.add_arg('-hs', '--s2s-hiddensize', type=int, default=30,
            help='size of the hidden layers and embeddings')
        argparser.add_arg('-nl', '--s2s-numlayers', type=int, default=1,
            help='number of hidden layers')
        argparser.add_arg('-lr', '--s2s-learningrate', type=float, default=0.01,
            help='learning rate')
        argparser.add_arg('--cuda', action='store_true',
            help='enable GPUs if available')

    @classmethod
    def lineToTensor(cls, line, tensor=None):
        """Turn a line into a vector of indices."""
        # preprocessing: remove UNK and EOS tokens
        line = line.replace(cls.UNK, '').replace(cls.EOS, '')
        if tensor is None:
            tensor = torch.LongTensor(len(line))
        elif tensor.size(0) < len(line):
            raise RuntimeError()
        for li, letter in enumerate(line):
            tensor[li] = cls.VOCAB.get(letter, cls.UNK)
        return tensor

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'Seq2Seq'
        self.num_layers = opt['s2s_numlayers']
        self.hidden_size = opt['s2s_hiddensize']
        self.learning_rate = opt['s2s_learningrate']
        self.cuda = opt['cuda']
        self.episode_done = True
        self.longest_label = 1


        self.criterion = nn.NLLLoss()
        self.lt = nn.Embedding(self.VOCAB_LEN, opt['s2s_hiddensize'], padding_idx=0)
        self.encoder = nn.LSTM(opt['s2s_hiddensize'], opt['s2s_hiddensize'], opt['s2s_numlayers'])
        self.decoder = nn.LSTM(opt['s2s_hiddensize'], opt['s2s_hiddensize'], opt['s2s_numlayers'])
        self.d2o = nn.Linear(opt['s2s_hiddensize'], self.VOCAB_LEN)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

        if self.cuda:
            self.cuda()

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
            raise RuntimeError()
        hidden = hidden.squeeze(0)
        scores = self.d2o(hidden)
        if drop:
            scores = self.dropout(scores)
        scores = self.softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores

    def zero_grad(self):
        self.lt.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.d2o.zero_grad()

    def update_params(self):
        for p in self.lt.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)
        for p in self.encoder.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)
        for p in self.decoder.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)
        for p in self.d2o.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)

    def init_zeros(self, bsz=1):
        return Variable(torch.zeros(self.num_layers, bsz, self.hidden_size))

    def init_ones(self, bsz=1):
        return Variable(torch.ones(self.num_layers, bsz, self.hidden_size))

    def init_rand(self, bsz=1):
        return Variable(torch.FloatTensor(self.num_layers, bsz, self.hidden_size).uniform_(0.05))

    def observe(self, observation):
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        obs = self.observation
        # print('----- observing:', obs['text'].replace('\n', '|'))

        # encode
        x = Variable(self.lineToTensor(obs['text']))
        xe = self.lt(x).unsqueeze(1)
        h0 = self.init_rand()
        c0 = self.init_zeros()
        _output, (hn, cn) = self.encoder(xe, (h0, c0))


        # decode
        x = self.EOS_TENSOR
        xe = self.lt(x).unsqueeze(1)
        cn = self.init_zeros()
        curr_output = None
        output_line = []

        if 'labels' in obs:
            self.zero_grad()
            # update model
            loss = 0
            targets = Variable(self.lineToTensor(random.choice(obs['labels']) + self.EOS))
            self.longest_label = max(self.longest_label, targets.size(0))
            for i in range(targets.size(0)):
                output, (hn, cn) = self.decoder(xe, (hn, cn))
                pred, scores = self.hidden_to_idx(output, drop=True)
                output_line.append(self.TOKENS[pred.data[0][0]])
                y = targets[i]
                loss += self.criterion(scores, y)
                # use the true token as the next input
                xe = self.lt(y).unsqueeze(1)
            loss.backward()
            self.update_params()
        else:
            while ((curr_output is None or curr_output != self.EOS_TENSOR) and
                    len(output_line) < self.longest_label):
                # keep predicting until end token is selected
                output, (hn, cn) = self.decoder(xe, (hn, cn))
                pred, scores = self.hidden_to_idx(output, drop=False)
                xe = self.lt(pred).unsqueeze(1)
                curr_output = pred
                output_line.append(self.TOKENS[pred.data[0][0]])

        action = {'id': self.getID()}
        action['text'] = ''.join(c for c in output_line if c != self.EOS)
        if 'labels' in obs:
            action['labels'] = obs['labels']
            action['metrics'] = {'loss': loss.data[0]}
        print(action)
        return action

    def update(self, xs, ys):
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).t()
        h0 = self.init_rand(batchsize)
        c0 = self.init_zeros(batchsize)
        _output, (hn, cn) = self.encoder(xes, (h0, c0))

        # start with EOS tensor for all
        x = self.EOS_TENSOR
        xe = self.lt(x).unsqueeze(1)
        xes = torch.cat([xe for _ in range(batchsize)], 1)

        cn = self.init_zeros(batchsize)
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0
        output_lines = [[] for _ in range(batchsize)]

        self.zero_grad()
        # update model
        loss = 0
        self.longest_label = max(self.longest_label, ys.size(1))
        for i in range(ys.size(1)):
            output, (hn, cn) = self.decoder(xes, (hn, cn))
            preds, scores = self.hidden_to_idx(output, drop=True)
            y = ys.select(1, i)
            loss += self.criterion(scores, y)
            # use the true token as the next input
            xes = self.lt(y).unsqueeze(0)
            for j in range(preds.size(0)):
                token = self.TOKENS[preds.data[j][0]]
                output_lines[j].append(token)

        loss.backward()
        self.update_params()

        return output_lines

    def predict(self, xs):
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).t()
        h0 = self.init_rand(batchsize)
        c0 = self.init_zeros(batchsize)
        _output, (hn, cn) = self.encoder(xes, (h0, c0))

        # start with EOS tensor for all
        x = self.EOS_TENSOR
        xe = self.lt(x).unsqueeze(1)
        xes = torch.cat([xe for _ in range(batchsize)], 1)

        cn = self.init_zeros(batchsize)
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0
        output_lines = [[] for _ in range(batchsize)]

        while(total_done < batchsize) and max_len < self.longest_label:
            output, (hn, cn) = self.decoder(xes, (hn, cn))
            preds, scores = self.hidden_to_idx(output, drop=False)
            xes = self.lt(preds.t())
            max_len += 1
            for i in range(preds.size(0)):
                if not done[i]:
                    token = self.TOKENS[preds.data[i][0]]
                    if token == self.EOS:
                        done[i] == True
                        total_done += 1
                    else:
                        output_lines[i].append(token)
        return output_lines


    def batchify(self, obs, cuda=False):
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        batchsize = len(exs)
        max_x_len = max([len(ex['text']) for ex in exs])
        xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
        for i, ex in enumerate(exs):
            self.lineToTensor(ex['text'], xs[i][-len(ex['text']):])
        if cuda:
            xs.pin_memory()
        xs = Variable(xs)

        ys = None
        if 'labels' in exs[0]:
            labels = [random.choice(ex['labels']) for ex in exs]
            max_y_len = max(len(y) for y in labels)
            ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(labels):
                self.lineToTensor(y, ys[i])
            if cuda:
                ys.pin_memory()
            ys = Variable(ys)

        return xs, ys, valid_inds

    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        xs, ys, valid_inds = self.batchify(observations, cuda=self.cuda)

        if len(xs) == 0:
            return batch_reply

        # Either train or predict
        if ys is not None:
            predictions = self.update(xs, ys)
        else:
            predictions = self.predict(xs)

        for i in range(len(predictions)):
            batch_reply[valid_inds[i]]['text'] = ''.join(c for c in predictions[i] if c != self.EOS)
        print(batch_reply)
        return batch_reply


def main():
    # Get command line arguments
    parser = ParlaiParser()
    Seq2SeqAgent.add_cmdline_args(parser)
    opt = parser.parse_args()

    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        print('[ Using CUDA (GPU %d) ]' % opt['gpu'])
        torch.cuda.set_device(opt['gpu'])

    agent = Seq2SeqAgent(opt)

    opt['datatype'] = 'train'
    world_train = create_task(opt, agent)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    start = time.time()
    # train / valid loop
    while True:
        print('[ training ]')
        for _ in range(20):  # train for a bit
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
