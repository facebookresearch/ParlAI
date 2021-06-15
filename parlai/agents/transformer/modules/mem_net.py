#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional, Tuple

from parlai.agents.transformer.modules import (
    create_embeddings,
    BasicAttention,
    TransformerEncoder,
    TransformerResponseWrapper,
)
from parlai.core.opt import Opt
from parlai.core.torch_agent import DictionaryAgent


class TransformerMemNetModel(nn.Module):
    """
    Model which takes context, memories, candidates and encodes them.
    """

    @classmethod
    def build_context_encoder(
        cls, opt, dictionary, embedding=None, padding_idx=None, reduction_type='mean'
    ):
        return cls.build_encoder(
            opt, dictionary, embedding, padding_idx, reduction_type
        )

    @classmethod
    def build_candidate_encoder(
        cls, opt, dictionary, embedding=None, padding_idx=None, reduction_type='mean'
    ):
        return cls.build_encoder(
            opt, dictionary, embedding, padding_idx, reduction_type
        )

    @classmethod
    def build_encoder(
        cls, opt, dictionary, embedding=None, padding_idx=None, reduction_type='mean'
    ):
        return TransformerEncoder(
            opt=opt,
            embedding=embedding,
            vocabulary_size=len(dictionary),
            padding_idx=padding_idx,
            reduction_type=reduction_type,
        )

    def __init__(self, opt: Opt, dictionary: DictionaryAgent):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]

        # set up embeddings
        self.embeddings = create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.share_word_embedding = opt.get('share_word_embeddings', True)
        if not self.share_word_embedding:
            self.cand_embeddings = create_embeddings(
                dictionary, opt['embedding_size'], self.pad_idx
            )

        if not opt.get('learn_embeddings'):
            self.embeddings.weight.requires_grad = False
            if not self.share_word_embedding:
                self.cand_embeddings.weight.requires_grad = False

        self.reduction_type = opt.get('reduction_type', 'mean')

        self.context_encoder = self.build_context_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=self.reduction_type,
        )

        if opt.get('share_encoders'):
            self.cand_encoder = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim
            )
        else:
            if not self.share_word_embedding:
                cand_embeddings = self.cand_embeddings
            else:
                cand_embeddings = self.embeddings
            self.cand_encoder = self.build_candidate_encoder(
                opt,
                dictionary,
                cand_embeddings,
                self.pad_idx,
                reduction_type=self.reduction_type,
            )

        # build memory encoder
        if opt.get('wrap_memory_encoder', False):
            self.memory_transformer = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim
            )
        else:
            self.memory_transformer = self.context_encoder

        self.attender = BasicAttention(
            dim=2, attn=opt['memory_attention'], residual=True
        )

    def encode_cand(self, words):
        """
        Encode the candidates.
        """
        if words is None:
            return None

        # flatten if there are many candidates
        if words.dim() == 3:
            oldshape = words.shape
            words = words.reshape(oldshape[0] * oldshape[1], oldshape[2])
        else:
            oldshape = None

        encoded = self.cand_encoder(words)

        if oldshape is not None:
            encoded = encoded.reshape(oldshape[0], oldshape[1], -1)

        return encoded

    def encode_context_memory(self, context_w, memories_w, context_segments=None):
        """
        Encode the context and memories.
        """
        # [batch, d]
        if context_w is None:
            # it's possible that only candidates were passed into the
            # forward function, return None here for LHS representation
            return None, None

        context_h = self.context_encoder(context_w, segments=context_segments)

        if memories_w is None:
            return [], context_h

        bsz = memories_w.size(0)
        memories_w = memories_w.view(-1, memories_w.size(-1))
        memories_h = self.memory_transformer(memories_w)
        memories_h = memories_h.view(bsz, -1, memories_h.size(-1))

        context_h = context_h.unsqueeze(1)
        context_h, weights = self.attender(context_h, memories_h)

        return weights, context_h

    def forward(
        self,
        xs: torch.LongTensor,
        mems: torch.LongTensor,
        cands: torch.LongTensor,
        context_segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] xs: input tokens IDs
        :param LongTensor[batch,num_mems,seqlen] mems: memory token IDs
        :param LongTensor[batch,num_cands,seqlen] cands: candidate token IDs
        :param LongTensor[batch,seqlen] context_segments: segment IDs for xs,
            used if n_segments is > 0 for the context encoder
        """
        # encode the context and memories together
        weights, context_h = self.encode_context_memory(
            xs, mems, context_segments=context_segments
        )
        # encode the candidates
        cands_h = self.encode_cand(cands)

        # possibly normalize the context and candidate representations
        if self.opt['normalize_sent_emb']:
            context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
            cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)

        return context_h, cands_h
