#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, opt, ntoken):
        super(RNNModel, self).__init__()
        self.opt = opt

        # set hyperparameters from opt
        rnn_type = opt['rnn_class']
        ninp = opt['embeddingsize']
        nhid = opt['hiddensize']
        nlayers = opt['numlayers']
        dropout = opt['dropout']
        tie_weights = opt['emb_tied']

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError(
                    "An invalid option for `--model` was supplied, options are "
                    "['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        #   Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize'
                )
            self.decoder.weight = self.encoder.weight

        # initialize the weights of the model
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden, no_pack=False):
        emb = self.drop(self.encoder(input))
        # if eval, pack padded sequence (we don't pack during training because
        # we have no padding in our input samples)
        if not self.training and not no_pack:
            emb_lens = [x for x in torch.sum((input > 0).int(), dim=0).data]
            emb_packed = pack_padded_sequence(emb, emb_lens, batch_first=False)
            packed_output, hidden = self.rnn(emb_packed, hidden)
            output, _ = pad_packed_sequence(packed_output, batch_first=False)
        else:
            output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
            )
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
