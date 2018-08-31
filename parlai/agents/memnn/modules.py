# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn

from functools import lru_cache


def opt_to_kwargs(opt):
    """Get kwargs for seq2seq from opt."""
    kwargs = {}
    for k in ['mem_size', 'time_features', 'position_encoding', 'hops'
              'dropout']:
        if k in opt:
            kwargs[k] = opt[k]
    return kwargs


class MemNN(nn.Module):
    """Memory Network module."""

    def __init__(
        self, num_features, embedding_size, hops=1,
        mem_size=32, time_features=False, position_encoding=False,
        dropout=0, padding_idx=0, use_cuda=False,
    ):
        """Initialize memnn model.

        See cmdline args in MemnnAgent for description of arguments.
        """
        super().__init__()

        # prepare features
        self.hops = hops

        # time features: we learn an embedding for each memory slot
        self.extra_features = 0
        if time_features:
            self.extra_features += mem_size
            self.time_features = torch.LongTensor(
                range(num_features, num_features + mem_size))

        def embedding(use_extra_feats=True):
            if use_extra_feats:
                return Embed(num_features + self.extra_features,
                             embedding_size,
                             position_encoding=position_encoding,
                             padding_idx=padding_idx)
            else:
                return Embed(num_features, embedding_size,
                             position_encoding=position_encoding,
                             padding_idx=padding_idx)

        self.dropout = nn.Dropout(dropout)
        self.query_lt = embedding()
        self.in_memory_lt = embedding()
        self.out_memory_lt = embedding()
        self.answer_embedder = embedding(use_extra_feats=False)
        self.memory_hop = Hop(embedding_size)

        if use_cuda:
            self.score.cuda()
            self.memory_hop.cuda()
        self.use_cuda = use_cuda

    def _score(self, output, cands):
        if isinstance(cands, torch.Tensor):
            return torch.matmul(output, cands.t())
        else:
            return torch.cat([
                torch.matmul(output[i], cands[i].t()).unsqueeze(0)
                for i in range(len(cands))], dim=0)

    def forward(self, xs, mems, cands=None):
        """One forward step.

        :param xs:       (bsz x seqlen) LongTensor queries to the model
        :param mems:     (bsz x num_mems x seqlen) LongTensor memories
        :param cands:

        :returns: scores
            scores contains the model's predicted scores.
                if cand_params is None, the candidates are the vocabulary;
                otherwise, these scores are over the candidates provided.
                (bsz x num_cands)
        """
        in_memory_embs = self.in_memory_lt(mems).transpose(1, 2)
        out_memory_embs = self.out_memory_lt(mems)
        states = self.query_lt(xs)

        for i in range(self.hops):
            states = self.memory_hop(states, in_memory_embs, out_memory_embs)

        if cands is not None:
            if isinstance(cands, torch.Tensor):
                cand_embs = self.answer_embedder(cands)
            else:
                cand_embs = [self.answer_embedder(cs) for cs in cands]
        else:
            cand_embs = self.answer_embedder.weight

        scores = self._score(states, cand_embs)
        return scores


class Embed(nn.Embedding):
    """Embed sequences for MemNN model.

    Applies Position Encoding if enabled and currently applies BOW sum.
    """

    def __init__(self, *args, position_encoding=False, reduction='sum',
                 **kwargs):
        """Initialize custom Embedding layer.

        :param position_encoding: apply positional encoding transformation
                                  on input sequences
        :param reduction:         reduction strategy to sequences, default 'sum'
        """
        self.position_encoding = position_encoding
        self.reduction = reduction
        super().__init__(*args, **kwargs)

    def _reduce(self, embs):
        # last dimension is embedding, do operation over dim before that
        if self.reduction == 'sum':
            return embs.sum(-2)
        elif self.reduction == 'mean':
            return embs.mean(-2)
        else:
            raise RuntimeError(
                'reduction method {} not supported'.format(self.reduction))

    def forward(self, input):
        """Return BOW embedding with PE reweighting if enabled.

        :param input: (bsz x seqlen) LongTensor

        :returns: (bsz x esz) FloatTensor
        """
        embs = super().forward(input)
        if self.position_encoding:
            if embs.dim() == 3:
                num_mems, seqlen, embdim = embs.size()
                pe = self.position_matrix(seqlen, embdim)
                for i in range(num_mems):
                    embs[i] *= pe
            else:
                raise RuntimeError(
                    'Input dim {} not supported with position encoding yet'
                    ''.format(input.dim()))
        return self._reduce(embs)

    @staticmethod
    @lru_cache(maxsize=128)
    def position_matrix(J, d):
        """Build matrix of position encoding coeffiencents.

        See https://papers.nips.cc/paper/5846-end-to-end-memory-networks,
        section 4.1 Model Details: Sentence Representation.

        :param J: number of words in the sequence
        :param d: dimension of the embedding

        :returns: Position Encoding matrix
        """
        m = torch.Tensor(J, d)
        for k in range(1, d + 1):
            for j in range(1, J + 1):
                m[j - 1, k - 1] = (1 - j / J) - (k / d) * (1 - 2 * j / J)
        return m


class Hop(nn.Module):
    """Memory Network hop outputs attention-weighted sum of memory embeddings.

    0) rotate the query embeddings
    1) compute the dot product between the input vector and each memory vector
    2) compute a softmax over the memory scores
    3) compute the weighted sum of the memory embeddings using the probabilities
    4) add the query embedding to the memory output and return the result
    """

    def __init__(self, embedding_size, rotate=True):
        """Initialize linear rotation."""
        super().__init__()
        if rotate:
            self.rotate = nn.Linear(embedding_size, embedding_size, bias=False)
        else:
            self.rotate = lambda x: x
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query_embs, in_mem_embs, out_mem_embs):
        """Compute MemNN Hop step.

        :param query_embs:   (bsz x esz) embedding of queries
        :param in_mem_embs:  bsz list of (num_mems x esz) embedding of memories
                             for activation
        :param out_mem_embs: bsz list of (num_mems x esz) embedding of memories
                             for outputs

        :returns: (bsz x esz) output state
        """
        # rotate query embeddings
        query_embs = self.rotate(query_embs)
        attn = torch.bmm(query_embs.unsqueeze(1), in_mem_embs).squeeze(1)
        probs = self.softmax(attn)
        memory_output = torch.bmm(probs.unsqueeze(1), out_mem_embs).squeeze(1)
        return memory_output + query_embs
