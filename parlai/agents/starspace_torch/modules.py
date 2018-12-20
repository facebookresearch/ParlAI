#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn


class Starspace(nn.Module):
    def __init__(self, opt, num_features, dict):
        super().__init__()
        self.lt = nn.Embedding(num_features, opt['embeddingsize'], 0,
                               sparse=True, max_norm=opt['embeddingnorm'])
        if not opt['tfidf']:
            dict = None
        self.encoder = Encoder(self.lt, dict)
        if not opt['share_embeddings']:
            self.lt2 = nn.Embedding(num_features, opt['embeddingsize'], 0,
                                    sparse=True, max_norm=opt['embeddingnorm'])
            self.encoder2 = Encoder(self.lt2, dict)
        else:
            self.encoder2 = self.encoder
        self.opt = opt
        self.lin = nn.Linear(opt['embeddingsize'], opt['embeddingsize'], bias=False)
        self.lins = 0
        if 'lins' in opt:
            self.lins = opt['lins']

    def forward(self, xs, ys=None, cands=None):
        xs_emb = self.encoder(xs)
        if self.lins > 0:
            xs_emb = self.lin(xs_emb)
        if ys is not None:
            # training includes the correct example first.
            ys_emb = self.encoder2(ys)
        if cands is not None:
            bsz = cands.size(0)
            cands = cands.view(-1, cands.size(-1))
            cands_emb = self.encoder2(cands)
            cands_emb = cands_emb.view(bsz, -1, cands_emb.size(-1))
            ys_emb = ys_emb.unsqueeze(1)
            ys_emb = torch.cat([ys_emb, cands_emb], dim=1)
        return xs_emb, ys_emb


class Encoder(nn.Module):
    def __init__(self, shared_lt, dict):
        super().__init__()
        self.lt = shared_lt
        if dict is not None:
            l = len(dict)
            freqs = torch.Tensor(l)
            for i in range(l):
                ind = dict.ind2tok[i]
                freq = dict.freq[ind]
                freqs[i] = 1.0 / (1.0 + math.log(1.0 + freq))
            self.freqs = freqs
        else:
            self.freqs = None

    def forward(self, xs):
        xs_emb = self.lt(xs)
        if self.freqs is not None:
            # TODO: fix this, it may or may not be broken
            # tfidf embeddings
            l = xs.size(1)
            w = torch.Tensor(l)
            for i in range(l):
                w[i] = self.freqs[xs.data[0][i]]
            w = w.mul(1 / w.norm())
            xs_emb = xs_emb.squeeze(0).t().matmul(w.unsqueeze(1)).t()
        else:
            # basic embeddings (faster)
            xs_emb = xs_emb.mean(1)
        return xs_emb
