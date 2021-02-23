#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements NN code for transformers.

Original paper: https://arxiv.org/abs/1706.03762. (Vaswani, 2017). The
`Annotated Transformer` (Rush, 2018) is an excellent reading guide which explains
much of the mechanics of the Transformer model
(http://nlp.seas.harvard.edu/2018/04/03/attention.html).

This module also supports special segments (ala BERT;
https://arxiv.org/abs/1810.04805), and a few different variations seen in the
literature (BERT and XLM; https://arxiv.org/abs/1901.07291).
"""

from typing import Dict

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from parlai.agents.transformer.modules.attention import BasicAttention
from parlai.agents.transformer.modules.decoder import TransformerDecoder
from parlai.agents.transformer.modules.encoder import TransformerEncoder
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.torch import neginf


def _create_embeddings(dictionary, embedding_size, padding_idx):
    """
    Create and initialize word embeddings.
    """
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


class TransformerMemNetModel(nn.Module):
    """
    Model which takes context, memories, candidates and encodes them.
    """

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

    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]

        # set up embeddings
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.share_word_embedding = opt.get('share_word_embeddings', True)
        if not self.share_word_embedding:
            self.cand_embeddings = _create_embeddings(
                dictionary, opt['embedding_size'], self.pad_idx
            )

        if not opt.get('learn_embeddings'):
            self.embeddings.weight.requires_grad = False
            if not self.share_word_embedding:
                self.cand_embeddings.weight.requires_grad = False

        self.reduction_type = opt.get('reduction_type', 'mean')

        self.context_encoder = self.build_encoder(
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
            self.cand_encoder = self.build_encoder(
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


class TransformerGeneratorModel(TorchGeneratorModel):
    """
    Implements a full generator model, with one encoder and one decoder.
    """

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

    @classmethod
    def build_decoder(cls, opt, dictionary, embedding=None):
        return TransformerDecoder(opt=opt, dictionary=dictionary, embedding=embedding)

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.encoder = self.build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction_type=None
        )
        self.decoder = self.build_decoder(
            opt, dictionary, self.embeddings, self.pad_idx
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output
