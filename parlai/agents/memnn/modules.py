#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from parlai.utils.torch import neginf
from functools import lru_cache


def opt_to_kwargs(opt):
    """
    Get kwargs for seq2seq from opt.
    """
    kwargs = {}
    for k in ['memsize', 'time_features', 'position_encoding', 'hops']:
        if k in opt:
            kwargs[k] = opt[k]
    return kwargs


class MemNN(nn.Module):
    """
    Memory Network module.
    """

    def __init__(
        self,
        num_features,
        embedding_size,
        hops=1,
        memsize=32,
        time_features=False,
        position_encoding=False,
        dropout=0,
        padding_idx=0,
    ):
        """
        Initialize memnn model.

        See cmdline args in MemnnAgent for description of arguments.
        """
        super().__init__()

        # prepare features
        self.hops = hops

        def embedding(use_extra_feats=True):
            return Embed(
                num_features,
                embedding_size,
                position_encoding=position_encoding,
                padding_idx=padding_idx,
            )

        # TODO: add token dropout?
        # TODO: add dropout
        # self.dropout = nn.Dropout(dropout)
        # TODO: support more weight tying options?
        self.query_lt = embedding()
        self.in_memory_lt = embedding()
        self.out_memory_lt = embedding()
        self.answer_embedder = embedding()
        self.memory_hop = Hop(embedding_size)

    def forward(self, xs, mems, cands=None, pad_mask=None):
        """
        One forward step.

        :param xs:
            (bsz x seqlen) LongTensor queries to the model

        :param mems:
            (bsz x num_mems x seqlen) LongTensor memories

        :param cands:
            (num_cands x seqlen) or (bsz x num_cands x seqlen)
            LongTensor with candidates to rank
        :param pad_mask:
            (bsz x num_mems) optional mask indicating which tokens
            correspond to padding

        :returns:
            scores contains the model's predicted scores.
            if cand_params is None, the candidates are the vocabulary;
            otherwise, these scores are over the candidates provided.
            (bsz x num_cands)
        """
        state = self.query_lt(xs)
        if mems is not None:
            # no memories available, `nomemnn` mode just uses query/ans embs
            in_memory_embs = self.in_memory_lt(mems).transpose(1, 2)
            out_memory_embs = self.out_memory_lt(mems)

            for _ in range(self.hops):
                state = self.memory_hop(
                    state, in_memory_embs, out_memory_embs, pad_mask
                )

        if cands is not None:
            # embed candidates
            cand_embs = self.answer_embedder(cands)
        else:
            # rank all possible tokens
            cand_embs = self.answer_embedder.weight

        return state, cand_embs


class Embed(nn.Embedding):
    """
    Embed sequences for MemNN model.

    Applies Position Encoding if enabled and currently applies BOW sum.
    """

    def __init__(self, *args, position_encoding=False, reduction='mean', **kwargs):
        """
        Initialize custom Embedding layer.

        :param position_encoding:
            apply positional encoding transformation on input sequences

        :param reduction:
            reduction strategy to sequences, default 'mean'
        """
        self.position_encoding = position_encoding
        self.reduction = reduction
        super().__init__(*args, **kwargs)

    def _reduce(self, embs, input):
        # last dimension is embedding, do operation over dim before that
        if self.reduction == 'sum':
            return embs.sum(-2)
        elif self.reduction == 'mean':
            # this is more fair than mean(-2) since mean includes null tokens
            sum = embs.sum(-2)
            lens = (
                input.ne(self.padding_idx).sum(-1).unsqueeze(-1).float().clamp_(min=1)
            )
            return sum / lens
        else:
            raise RuntimeError(
                'reduction method {} not supported'.format(self.reduction)
            )

    def forward(self, input):
        """
        Return BOW embedding with PE reweighting if enabled.

        :param input:
            (bsz x seqlen) LongTensor

        :returns:
            (bsz x esz) FloatTensor
        """
        embs = super().forward(input)
        if self.position_encoding:
            if embs.dim() == 3:
                num_mems, seqlen, embdim = embs.size()
                pe = self.position_matrix(seqlen, embdim, embs.is_cuda)
                for i in range(num_mems):
                    embs[i] *= pe
            else:
                bsz, num_mems, seqlen, embdim = embs.size()
                pe = self.position_matrix(seqlen, embdim, embs.is_cuda)
                for i in range(num_mems):
                    embs[:, i] *= pe
        return self._reduce(embs, input)

    @staticmethod
    @lru_cache(maxsize=128)
    def position_matrix(J, d, use_cuda):
        """
        Build matrix of position encoding coeffiencents.

        See https://papers.nips.cc/paper/5846-end-to-end-memory-networks,
        section 4.1 Model Details: Sentence Representation.

        :param J:
            number of words in the sequence

        :param d:
            dimension of the embedding

        :returns:
            Position Encoding matrix
        """
        m = torch.Tensor(J, d)
        for k in range(1, d + 1):
            for j in range(1, J + 1):
                m[j - 1, k - 1] = (1 - j / J) - (k / d) * (1 - 2 * j / J)
        if use_cuda:
            m = m.cuda()
        return m


class Hop(nn.Module):
    """
    Memory Network hop outputs attention-weighted sum of memory embeddings.

    0) rotate the query embeddings 1) compute the dot product between the input vector
    and each memory vector 2) compute a softmax over the memory scores 3) compute the
    weighted sum of the memory embeddings using the probabilities 4) add the query
    embedding to the memory output and return the result
    """

    def __init__(self, embedding_size, rotate=True):
        """
        Initialize linear rotation.
        """
        super().__init__()
        if rotate:
            self.rotate = nn.Linear(embedding_size, embedding_size, bias=False)
        else:
            self.rotate = lambda x: x
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query_embs, in_mem_embs, out_mem_embs, pad_mask):
        """
        Compute MemNN Hop step.

        :param query_embs:
            (bsz x esz) embedding of queries

        :param in_mem_embs:
            bsz list of (num_mems x esz) embedding of memories for activation

        :param out_mem_embs:
            bsz list of (num_mems x esz) embedding of memories for outputs

        :param pad_mask
            (bsz x num_mems) optional mask indicating which tokens correspond to
            padding

        :returns:
            (bsz x esz) output state
        """
        # rotate query embeddings
        attn = torch.bmm(query_embs.unsqueeze(1), in_mem_embs).squeeze(1)
        if pad_mask is not None:
            attn[pad_mask] = neginf(attn.dtype)
        probs = self.softmax(attn)
        memory_output = torch.bmm(probs.unsqueeze(1), out_mem_embs).squeeze(1)
        output = memory_output + self.rotate(query_embs)
        return output
