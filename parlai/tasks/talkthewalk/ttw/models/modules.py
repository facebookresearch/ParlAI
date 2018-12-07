# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

from functools import reduce
from torch.autograd import Variable

class CBoW(nn.Module):

    def __init__(self, num_tokens, emb_size, init_std=1, padding_idx=None):
        super(CBoW, self).__init__()
        self.emb_fn = nn.Embedding(num_tokens, emb_size, padding_idx=padding_idx)
        if init_std != 1.0:
            self.emb_fn.weight.data.normal_(0.0, init_std)
        self.emb_size = emb_size

    def forward(self, x):
        in_shape = x.size()
        num_elem = reduce(operator.mul, in_shape)
        flat_x = x.contiguous().view(num_elem)
        flat_emb = self.emb_fn.forward(flat_x)
        emb = flat_emb.view(*(in_shape+(self.emb_size,)))
        return emb.sum(dim=-2)


class MASC(nn.Module):

    def __init__(self, hidden_sz):
        super(MASC, self).__init__()
        self.conv_weight = nn.Parameter(torch.FloatTensor(
            hidden_sz, hidden_sz, 3, 3))
        std = 1.0 / (hidden_sz * 9)
        self.conv_weight.data.uniform_(-std, std)

    def forward(self, inp, action_out, current_step=None, Ts=None):
        batch_size = inp.size(0)
        out = inp.clone().zero_()

        for i in range(batch_size):
            if Ts is None or current_step <= Ts[i]:
                selected_inp = inp[i, :, :, :].unsqueeze(0)
                mask = F.softmax(action_out[i], dim=0).view(1, 1, 3, 3)
                weight = mask * self.conv_weight
                out[i, :, :, :] = F.conv2d(selected_inp, weight, padding=1).squeeze(0)
        return out


class NoMASC(MASC):

    def forward(self, input):
        mask = torch.FloatTensor(1, 1, 3, 3).zero_()
        mask[0, 0, 0, 1] = 1.0
        mask[0, 0, 1, 0] = 1.0
        mask[0, 0, 2, 1] = 1.0
        mask[0, 0, 1, 2] = 1.0

        mask = Variable(mask)
        if input.is_cuda:
            mask = mask.cuda()

        weight = self.conv_weight * mask
        return F.conv2d(input, weight, padding=1)


class ControlStep(nn.Module):

    def __init__(self, hidden_sz):
        super(ControlStep, self).__init__()
        self.control_updater = nn.Linear(2 * hidden_sz, hidden_sz)
        self.hop_fn = AttentionHop()

    def forward(self, inp_seq, mask, query):
        extracted_msg = self.hop_fn.forward(inp_seq, mask, query)
        conc_emb = torch.cat([query, extracted_msg], 1)
        control_emb = self.control_updater.forward(conc_emb)
        return extracted_msg, control_emb


class AttentionHop(nn.Module):

    def forward(self, inp_seq, mask, query):
        score = torch.bmm(inp_seq, query.unsqueeze(-1)).squeeze(-1)
        score = score - 1e30 * (1.0 - mask)
        att_score = F.softmax(score, dim=-1)
        extracted_msg = torch.bmm(att_score.unsqueeze(1), inp_seq).squeeze(1)
        return extracted_msg

class GRUEncoder(nn.Module):

    def __init__(self, emb_sz, hid_sz, num_emb, cbow=False):
        super(GRUEncoder, self).__init__()
        self.emb_sz = emb_sz
        self.hid_sz = hid_sz
        self.num_emb = num_emb
        self.cbow = cbow

        if cbow:
            self.emb_fn = CBoW(num_emb, emb_sz, init_std=0.1, padding_idx=0)
        else:
            self.emb_fn = nn.Embedding(num_emb, emb_sz, padding_idx=0)

        self.encoder = nn.GRU(emb_sz, hid_sz, batch_first=True)

    def forward(self, inp, seq_len):
        inp_emb = self.emb_fn(inp)
        states, _ = self.encoder(inp_emb)

        return self.get_last_state(states, seq_len)

    def get_last_state(self, states, seq_lens):
        batch_size = seq_lens.size(0)
        # append zero vector as first hidden state
        first_h = Variable(torch.FloatTensor(batch_size, 1, states.size(2)).zero_())
        if states.is_cuda:
            first_h = first_h.cuda()
        states = torch.cat([first_h, states], 1)

        return states[torch.arange(batch_size).long(), seq_lens, :]
