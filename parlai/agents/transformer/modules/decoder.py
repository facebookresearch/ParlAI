#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer decoder implementations.
"""

from __future__ import annotations
from abc import ABC
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from parlai.agents.transformer.modules import (
    create_position_codes,
    get_n_positions_from_options,
    LAYER_NORM_EPS,
    MultiHeadAttention,
    TransformerFFN,
)
from parlai.agents.transformer.modules.modular import swappable
from parlai.core.opt import Opt
from parlai.core.params import default
from parlai.core.torch_agent import DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.torch import PipelineHelper
from parlai.utils.fsdp import fsdp_wrap
from parlai.nn.checkpoint import checkpoint_wrapper


################
# BASE CLASSES #
################

DecoderIncrState = Dict[int, Dict[str, Dict[str, torch.Tensor]]]


class BaseTransformerDecoder(nn.Module, ABC):
    """
    Implements functionality common to all transformer decoder variants. Not intended to
    be instantiated directly.

    For a (Vaswani 2017) style encoder-decoder transformer, use ``TransformerDecoder``. For a GPT-style decoder-only transformer, use ``TransformerDecoderOnly``.

    Subclasses are required to implement ``forward``. In your ``forward`` implementation, you can call ``forward_embedding`` to get embeddings for the input tokens and ``forward_layers`` to pass those embeddings sequentially through each layer.

    Subclasses can optionally override ``__init__``, ``build_layer``, and
    ``build_layers`` to customize subcomponents. In particular, ``build_layer`` can be used to instantiate heterogeneous layers (e.g. every other layer being a different type).
    """

    def __init__(
        self,
        opt: Opt,
        embedding: nn.Embedding,
        dictionary: DictionaryAgent,
        n_positions: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]

        self.embedding_size = opt['embedding_size']
        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.n_heads = opt['n_heads']
        self.dim = self.embedding_size
        self.activation = opt.get('activation', 'relu')
        self.variant = opt.get('variant', 'aiayn')

        self.embeddings_scale = opt.get('embeddings_scale', True)
        self.dropout = nn.Dropout(p=opt.get('dropout', 0.0))  # --dropout

        self.n_positions = default(n_positions, get_n_positions_from_options(opt))
        self.out_dim = self.embedding_size
        assert (
            self.embedding_size % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
            if self.variant == 'xlm':
                warn_once(
                    'DEPRECATED: XLM should only be used for backwards compatibility, '
                    'as it involves a less-stable layernorm operation.'
                )
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.embedding_size,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size**-0.5
            )

        # build the model
        self.layers = self.build_layers()

    def build_layers(self) -> nn.ModuleList:
        """
        Instantiates all layers. Called only once during __init__.

        Additional setup common to all layers, such as checkpoint wrapping, can be done
        here.
        """
        layers = nn.ModuleList()
        for i in range(self.n_layers):
            layer = self.build_layer(index=i)
            if self.opt.get('checkpoint_activations'):
                layer = checkpoint_wrapper(layer)
            layers.append(fsdp_wrap(layer))  # type: ignore
        return layers

    def build_layer(self, index: int) -> BaseTransformerDecoderLayer:
        """
        Instantiate a single layer. Called n_layers times during __init__.

        :param int index:
            Index of current layer.
        """
        return BaseTransformerDecoderLayer(  # type: ignore
            self.opt,
            attention_dropout=self.opt.get('attention_dropout', 0.0),
            relu_dropout=self.opt.get('relu_dropout', 0.0),
            dropout=self.opt.get('dropout', 0.0),
            activation=self.activation,
            variant=self.variant,
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        raise NotImplementedError

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch, seqlen] input:
            The target input IDs
        :param LongTensor[batch, seqlen] positions:
            Positions for input IDs. If None, computes defaults.
        :param LongTensor[batch, seqlen] segments:
            Segment IDs for extra embedding features. If None, not used.

        :return (tensor, mask):
            embedded input and mask
        """
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = self.norm_embeddings(tensor)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        return tensor

    def forward_layers(
        self, tensor: torch.Tensor, *extra_args, incr_state: DecoderIncrState, **kwargs
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param extra_args:
            any number of positional arguments to be passed to each layer
        :param incr_state:
            Dict mapping layer_idx to incremental state
        :param kwargs:
            any number of keyword (named) arguments to be passed to each layer

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, *extra_args, incr_state=incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                tensor, new_incr_state[idx] = layer(
                    tensor, *extra_args, incr_state=incr_state.get(idx), **kwargs
                )

        return tensor, new_incr_state

    def _apply_model_parallel(
        self, tensor: torch.Tensor, *extra_args, incr_state: DecoderIncrState
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split((tensor, *extra_args, incr_state))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        new_incr_state = {i: [] for i, _ in enumerate(self.layers)}

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, *s_extra_args, s_incr_state = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor, nis = self.layers[layer_no](
                    s_tensor, *s_extra_args, incr_state=s_incr_state.get(layer_no)
                )
                new_incr_state[layer_no].append(nis)
            # don't move incr state, it's always on the correct device
            s_layer_args = PipelineHelper.chunk_to(
                (s_tensor, *s_extra_args), next_device
            )
            chunks[chunk_idx] = (*s_layer_args, s_incr_state)

        tensor_out = PipelineHelper.join([c[0] for c in chunks])
        new_incr_state = {
            layer_no: PipelineHelper.join(pieces)
            for layer_no, pieces in new_incr_state.items()
        }

        return tensor_out, new_incr_state


DecoderLayerIncrState = Dict[str, Dict[str, torch.Tensor]]


class BaseTransformerDecoderLayer(nn.Module, ABC):
    """
    Implements functionality common to all transformer decoder layer variants. Subclass
    this if you'd like to modify the behavior of any layer in a transformer decoder.

    While this code is functional, it is not intended to be instantiated directly. If
    this functionality is desired as-is, use TransformerDecoderOnlyLayer instead to gain
    the ability to swap self-attention and feedforward classes at instantiation.
    """

    def __init__(
        self,
        opt: Opt,
        n_heads: int = None,
        embedding_size: int = None,
        ffn_size: int = None,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__()

        n_heads = default(n_heads, opt['n_heads'])
        embedding_size = default(embedding_size, opt['embedding_size'])
        ffn_size = default(ffn_size, opt['ffn_size'])

        self.opt = opt
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = self.build_self_attention(
            n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )
        self.norm1 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = self.build_feedforward(
            dim=embedding_size,
            dim_hidden=ffn_size,
            relu_dropout=relu_dropout,
            activation=activation,
        )
        self.norm3 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def build_self_attention(
        self, n_heads: int = None, dim: int = None, dropout: float = 0
    ) -> MultiHeadAttention:
        return MultiHeadAttention(
            opt=self.opt, n_heads=n_heads, dim=dim, dropout=dropout
        )

    def build_feedforward(
        self,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
    ) -> TransformerFFN:
        return TransformerFFN(
            opt=self.opt,
            dim=dim,
            dim_hidden=dim_hidden,
            relu_dropout=relu_dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *extra_args,
        incr_state: Optional[DecoderLayerIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderLayerIncrState]:
        """
        Forward pass.

        The incremental state is a dict with values for self-attention states.
        """
        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm1(x)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm1(x)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm3(x)
        x = self.ffn(x, **kwargs)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm3(x)

        return x, {'self_attn': final_self_attn_incr_state}

    def reorder_incremental_state(
        self, incremental_state: DecoderLayerIncrState, inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {'self_attn': self.self_attention}
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


###########################
# ENCODER-DECODER MODULES #
###########################


@swappable(
    self_attention=MultiHeadAttention,
    encoder_attention=MultiHeadAttention,
    feedforward=TransformerFFN,
)
class TransformerDecoderLayer(BaseTransformerDecoderLayer):
    """
    Implements a single Transformer decoder layer with cross (encoder) attention as in.

    [Vaswani, 2017](https://arxiv.org/abs/1706.03762).

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a causal (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        opt: Opt,
        n_heads: int = None,
        embedding_size: int = None,
        ffn_size: int = None,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__(
            opt=opt,
            n_heads=n_heads,
            embedding_size=embedding_size,
            ffn_size=ffn_size,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            dropout=dropout,
            activation=activation,
            variant=variant,
            **kwargs,
        )

        n_heads = default(n_heads, opt['n_heads'])
        embedding_size = default(embedding_size, opt['embedding_size'])

        self.encoder_attention = self.swappables.encoder_attention(  # type: ignore
            opt=self.opt, n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )
        self.norm2 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def build_self_attention(
        self, n_heads: int = None, dim: int = None, dropout: float = 0
    ) -> MultiHeadAttention:
        """
        Overridden to allow swapping out of the attention class at instantiation.
        """
        return self.swappables.self_attention(  # type: ignore
            opt=self.opt, n_heads=n_heads, dim=dim, dropout=dropout
        )

    def build_feedforward(
        self,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
    ) -> TransformerFFN:
        """
        Overridden to allow swapping out of the feedforward class at instantiation.
        """
        return self.swappables.feedforward(  # type: ignore
            opt=self.opt,
            dim=dim,
            dim_hidden=dim_hidden,
            relu_dropout=relu_dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Optional[DecoderLayerIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderLayerIncrState]:
        """
        Forward pass.

        The incremental state is a dict with values for self- and encoder-attention
        states.
        """

        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm1(x)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm1(x)

        residual = x
        # encoder_attn_layer_norm norm 2
        if self.variant == 'prelayernorm':
            x = self.norm2(x)
        x, final_encoder_attn_incr_state = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            incr_state=incr_state.get('encoder_attn'),
            static_kv=True,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm2(x)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm3(x)
        x = self.ffn(x, **kwargs)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm3(x)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_encoder_attn_incr_state,
        }
        return x, new_incr_state

    def reorder_incremental_state(
        self, incremental_state: DecoderLayerIncrState, inds: torch.Tensor
    ) -> DecoderLayerIncrState:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'encoder_attn': self.encoder_attention,
        }
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }


@swappable(layer=TransformerDecoderLayer)
class TransformerDecoder(BaseTransformerDecoder):
    """
    Transformer Decoder module.

    For documentation on parameters that are take directly from opt,
    see parlai/agents/transformer/transformer.py

    :param opt: ParlAI-parsed options.
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def build_layer(self, index: int) -> BaseTransformerDecoderLayer:
        """
        Instantiate a single layer. Called n_layers times during __init__.

        Overridden to allow swapping out of the layer class at instantiation.

        :param int index:
            Index of current layer.
        """
        return self.swappables.layer(  # type: ignore
            self.opt,
            attention_dropout=self.opt.get('attention_dropout', 0.0),
            relu_dropout=self.opt.get('relu_dropout', 0.0),
            dropout=self.opt.get('dropout', 0.0),
            activation=self.activation,
            variant=self.variant,
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=input.device
        ).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions, **kwargs)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, encoder_output, encoder_mask, incr_state=incr_state, **kwargs
        )

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        return tensor, new_incr_state


########################
# DECODER-ONLY MODULES #
########################


@swappable(self_attention=MultiHeadAttention, feedforward=TransformerFFN)
class TransformerDecoderOnlyLayer(BaseTransformerDecoderLayer):
    """
    Implements a single Transformer decoder layer without attending to the encoder
    states.

    Decoder layers are similar to encoder layers but self-attention is limited in a
    causal (auto-regressive) manner.
    """

    def build_self_attention(
        self, n_heads: int = None, dim: int = None, dropout: float = 0
    ) -> MultiHeadAttention:
        """
        Overridden to allow swapping out of the self attention class at instantiation.
        """
        return self.swappables.self_attention(  # type: ignore
            opt=self.opt, n_heads=n_heads, dim=dim, dropout=dropout
        )

    def build_feedforward(
        self,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
    ) -> TransformerFFN:
        """
        Overridden to allow swapping out of the feedforward class at instantiation.
        """
        return self.swappables.feedforward(  # type: ignore
            opt=self.opt,
            dim=dim,
            dim_hidden=dim_hidden,
            relu_dropout=relu_dropout,
            activation=activation,
        )


@swappable(layer=TransformerDecoderOnlyLayer)
class TransformerDecoderOnly(BaseTransformerDecoder):
    """
    Transformer Decoder module for decoder-only architecture.

    Intended to be used with a PassThroughEncoder, which will pass the context to this decoder unchanged.

    For documentation on parameters that are taken directly from opt,
    see parlai/agents/transformer/transformer.py
    """

    def build_layer(self, index: int) -> BaseTransformerDecoderLayer:
        """
        Instantiate a single layer. Called n_layers times during __init__.

        Overridden to allow swapping out of the layer class at instantiation.

        :param int index:
            Index of current layer.
        """
        return self.swappables.layer(  # type: ignore
            self.opt,
            attention_dropout=self.opt.get('attention_dropout', 0.0),
            relu_dropout=self.opt.get('relu_dropout', 0.0),
            dropout=self.opt.get('dropout', 0.0),
            activation=self.activation,
            variant=self.variant,
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, ...],
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        context, *_ = encoder_state

        is_incr_decoding = incr_state is not None

        full_sequence = torch.cat([context, input], dim=-1)
        attention_mask = full_sequence != self.pad_idx
        position_ids = (attention_mask.cumsum(dim=-1, dtype=torch.int64) - 1).clamp_(
            min=0
        )
        if not is_incr_decoding:
            # forced decoding: concatenate the context with the labels
            model_input = full_sequence
        else:
            # generating with continuation: get the last token input
            model_input = input[:, -1:]
            position_ids = position_ids[:, -1:]

        tensor = self.forward_embedding(model_input, position_ids, **kwargs)

        tensor = self.dropout(tensor)  # --dropout

        if incr_state is None:
            incr_state = {}
        tensor, new_incr_state = self.forward_layers(
            tensor, incr_state=incr_state, **kwargs
        )

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        if not is_incr_decoding:
            # In case context has been prepended to the input, remove it from output.
            # Start token is prepended to input in TorchGeneratorModel.decode_forced,
            # so model_input is actually `context + [start_tkn] + label[:-1]`.
            # Since model predicts next token at each timestep, prediction starts at
            # the position where start_tkn was.
            tensor = tensor[:, context.size(1) :, :]

        return tensor, new_incr_state
