# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Helper functions for module initialization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F


def init_rnn(rnn, init_range, weights=None, biases=None):
    """Initializes RNN uniformly."""
    weights = weights or ['weight_ih_l0', 'weight_hh_l0']
    biases = biases or ['bias_ih_l0', 'bias_hh_l0']
    # Init weights
    for w in weights:
        rnn._parameters[w].data.uniform_(-init_range, init_range)
    # Init biases
    for b in biases:
        rnn._parameters[b].data.fill_(0)


def init_rnn_cell(rnn, init_range):
    """Initializes RNNCell uniformly."""
    init_rnn(rnn, init_range, ['weight_ih', 'weight_hh'], ['bias_ih', 'bias_hh'])


def init_cont(cont, init_range):
    """Initializes a container uniformly."""
    for m in cont:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)


class CudaModule(nn.Module):
    """A helper to run a module on a particular device using CUDA."""
    def __init__(self, device_id):
        super(CudaModule, self).__init__()
        self.device_id = device_id

    def to_device(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m


class RnnContextEncoder(CudaModule):
    """A module that encodes dialogues context using an RNN."""
    def __init__(self, n, k, nembed, nhid, init_range, device_id):
        super(RnnContextEncoder, self).__init__(device_id)
        self.nhid = nhid

        # use the same embedding for counts and values
        self.embeder = nn.Embedding(n, nembed)
        # an RNN to encode a sequence of counts and values
        self.encoder = nn.GRU(
            input_size=nembed,
            hidden_size=nhid,
            bias=True)

        self.embeder.weight.data.uniform_(-init_range, init_range)
        init_rnn(self.encoder, init_range)

    def forward(self, ctx):
        ctx_h = self.to_device(torch.zeros(1, ctx.size(1), self.nhid))
        # create embedding
        ctx_emb = self.embeder(ctx)
        # run it through the RNN to get a hidden representation of the context
        _, ctx_h = self.encoder(ctx_emb, Variable(ctx_h))
        return ctx_h


class MlpContextEncoder(CudaModule):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, n, k, nembed, nhid, init_range, device_id):
        super(MlpContextEncoder, self).__init__(device_id)

        # create separate embedding for counts and values
        self.cnt_enc = nn.Embedding(n, nembed)
        self.val_enc = nn.Embedding(n, nembed)

        self.encoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(k * nembed, nhid)
        )

        self.cnt_enc.weight.data.uniform_(-init_range, init_range)
        self.val_enc.weight.data.uniform_(-init_range, init_range)
        init_cont(self.encoder, init_range)

    def forward(self, ctx):
        idx = np.arange(ctx.size(0) // 2)
        # extract counts and values
        cnt_idx = Variable(self.to_device(torch.from_numpy(2 * idx + 0)))
        val_idx = Variable(self.to_device(torch.from_numpy(2 * idx + 1)))

        cnt = ctx.index_select(0, cnt_idx)
        val = ctx.index_select(0, val_idx)

        # embed counts and values
        cnt_emb = self.cnt_enc(cnt)
        val_emb = self.val_enc(val)

        # element wise multiplication to get a hidden state
        h = torch.mul(cnt_emb, val_emb)
        # run the hidden state through the MLP
        h = h.transpose(0, 1).contiguous().view(ctx.size(1), -1)
        ctx_h = self.encoder(h).unsqueeze(0)
        return ctx_h
