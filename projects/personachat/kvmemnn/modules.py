#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Kvmemnn(nn.Module):
    def __init__(self, opt, num_features, dict):
        super().__init__()
        self.lt = nn.Embedding(
            num_features,
            opt['embeddingsize'],
            0,
            sparse=True,
            max_norm=opt['embeddingnorm'],
        )
        if not opt['tfidf']:
            dict = None
        self.encoder = Encoder(self.lt, dict)
        if not opt['share_embeddings']:
            self.lt2 = nn.Embedding(
                num_features,
                opt['embeddingsize'],
                0,
                sparse=True,
                max_norm=opt['embeddingnorm'],
            )
            self.encoder2 = Encoder(self.lt2, dict)
        else:
            self.encoder2 = self.encoder
        self.opt = opt
        self.softmax = nn.Softmax(dim=1)
        self.cosine = nn.CosineSimilarity()

        self.lin1 = nn.Linear(opt['embeddingsize'], opt['embeddingsize'], bias=False)
        self.lin2 = nn.Linear(opt['embeddingsize'], opt['embeddingsize'], bias=False)
        self.hops = 1
        self.lins = 0
        if 'hops' in opt:
            self.hops = opt['hops']
        if 'lins' in opt:
            self.lins = opt['lins']
        self.cosineEmbedding = True
        if opt['loss'] == 'nll':
            self.cosineEmbedding = False

    def forward(self, xs, mems, ys=None, cands=None):
        xs_enc = []
        xs_emb = self.encoder(xs)

        if len(mems) > 0 and self.hops > 0:
            mem_enc = []
            for m in mems:
                mem_enc.append(self.encoder(m))
            mem_enc.append(xs_emb)
            mems_enc = torch.cat(mem_enc)
            self.layer_mems = mems
            layer2 = self.cosine(xs_emb, mems_enc).unsqueeze(0)
            self.layer2 = layer2
            layer3 = self.softmax(layer2)
            self.layer3 = layer3
            lhs_emb = torch.mm(layer3, mems_enc)

            if self.lins > 0:
                lhs_emb = self.lin1(lhs_emb)
            if self.hops > 1:
                layer4 = self.cosine(lhs_emb, mems_enc).unsqueeze(0)
                layer5 = self.softmax(layer4)
                self.layer5 = layer5
                lhs_emb = torch.mm(layer5, mems_enc)
                if self.lins > 1:
                    lhs_emb = self.lin2(lhs_emb)
        else:
            if self.lins > 0:
                lhs_emb = self.lin1(xs_emb)
            else:
                lhs_emb = xs_emb
        if ys is not None:
            # training
            if self.cosineEmbedding:
                ys_enc = []
                xs_enc.append(lhs_emb)
                ys_enc.append(self.encoder2(ys))
                for c in cands:
                    xs_enc.append(lhs_emb)
                    c_emb = self.encoder2(c)
                    ys_enc.append(c_emb)
            else:
                xs_enc.append(lhs_emb.dot(self.encoder2(ys)))
                for c in cands:
                    c_emb = self.encoder2(c)
                    xs_enc.append(lhs_emb.dot(c_emb))
        else:
            # test
            if self.cosineEmbedding:
                ys_enc = []
                for c in cands:
                    xs_enc.append(lhs_emb)
                    c_emb = self.encoder2(c)
                    ys_enc.append(c_emb)
            else:
                for c in cands:
                    c_emb = self.encoder2(c)
                    xs_enc.append(lhs_emb.dot(c_emb))
        if self.cosineEmbedding:
            return torch.cat(xs_enc), torch.cat(ys_enc)
        else:
            return torch.cat(xs_enc)


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
            # tfidf embeddings
            l = xs.size(1)
            w = Variable(torch.Tensor(l))
            for i in range(l):
                w[i] = self.freqs[xs.data[0][i]]
            w = w.mul(1 / w.norm())
            xs_emb = xs_emb.squeeze(0).t().matmul(w.unsqueeze(1)).t()
        else:
            # basic embeddings (faster)
            xs_emb = xs_emb.mean(1)
        return xs_emb
