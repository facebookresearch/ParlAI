#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from parlai.agents.transformer.functions import create_embeddings
from parlai.agents.transformer.modules.attention import BasicAttention
from parlai.agents.transformer.modules.encoder import TransformerEncoder


def get_n_positions_from_options(opt):
    """
    Determine n_positions from options dict.
    """
    if opt.get('n_positions'):
        # if the number of positions is explicitly provided, use that
        n_positions = opt['n_positions']
    else:
        # else, use the worst case from truncate
        n_positions = max(
            opt.get('truncate') or 0,
            opt.get('text_truncate') or 0,
            opt.get('label_truncate') or 0,
        )
        if n_positions == 0:
            n_positions = 1024
    return n_positions


class TransformerMemNetModel(nn.Module):
    """
    Model which takes context, memories, candidates and encodes them.
    """

    @classmethod
    def build_encoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        n_positions=1024,
        n_segments=0,
    ):
        n_layers = (
            opt['n_encoder_layers']
            if opt.get('n_encoder_layers', -1) > 0
            else opt['n_layers']
        )
        return TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=n_layers,
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dictionary),
            embedding=embedding,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=padding_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            reduction_type=reduction_type,
            n_positions=n_positions,
            n_segments=n_segments,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )

    def __init__(self, opt, dictionary):
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

        n_positions = get_n_positions_from_options(opt)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.reduction_type = opt.get('reduction_type', 'mean')
        self.n_segments = opt.get('n_segments', 0)

        self.context_encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=self.reduction_type,
            n_positions=n_positions,
            n_segments=self.n_segments,
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
            self.cand_encoder = self.build_encoder(
                opt,
                dictionary,
                cand_embeddings,
                self.pad_idx,
                n_positions=n_positions,
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

    def forward(self, xs, mems, cands, context_segments=None):
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


class TransformerResponseWrapper(nn.Module):
    """
    Wrap transformer response.

    Pushes input through transformer and MLP.
    """

    def __init__(self, transformer, hdim):
        super(TransformerResponseWrapper, self).__init__()
        dim = transformer.out_dim
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.ReLU(),  # TODO: should this also be gelu?
            nn.Linear(hdim, dim),
        )

    def forward(self, *args):
        """
        Forward pass.
        """
        return self.mlp(self.transformer(*args))


class TransformerLinearWrapper(nn.Module):
    """
    Wrap a transformer in a linear layer.
    """

    def __init__(self, transformer, output_dim):
        super().__init__()
        self.transformer = transformer
        input_dim = transformer.out_dim
        self.additional_linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, *args):
        """
        Forward pass.

        Apply transformer, then additional linear layer.
        """
        context_h = self.transformer(*args)
        return self.additional_linear_layer(context_h)
