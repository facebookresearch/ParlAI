#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch
from typing import Dict, Optional, Tuple
import torch.nn as nn

from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerGeneratorModel,
)

from parlai.agents.transformer.modules import (
    create_position_codes,
    get_n_positions_from_options,
    LAYER_NORM_EPS,
)

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.misc import warn_once
import torch.nn.functional as F

from torch.nn import LayerNorm

###########################################
#         Hash Ladder Transformer         #
###########################################

"""
Use with, e.g.:

parlai train_model -m projects.params_vs_compute.hash_ladder.hash_ladder:HashLadderAgent -t convai2:normalized -mf /tmp/model_file --ladder-size 1 --hash-size 32 --hash-layer 1
"""


class HashLadderAgent(TransformerGeneratorAgent):
    """
    Simple implementation of Hash Layers and the Ladder model from the following papers:

    https://arxiv.org/abs/2106.04426
    https://arxiv.org/abs/2106.04279

    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        # Add transformer args.
        parser.add_argument(
            '--ladder-size',
            type=int,
            default=1,
            help='Number of ladder steps, default is not to use ladder.',
        )
        parser.add_argument(
            '--hash-size', type=int, default=32, help='Number of hash bins.'
        )
        parser.add_argument(
            '--hash-layer',
            type=int,
            default=7,
            help='Layer number the Hash Layer appears on.',
        )
        return parser

    def build_model(self, states=None):
        wrapped_class = TransformerGeneratorModel.with_components(decoder=Decoder)
        return wrapped_class(self.opt, self.dict)


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm.
    """
    is_cpu = tensor.device == 'cpu' or tensor.device.type == 'cpu'
    return norm_layer(tensor)


class Decoder(TransformerDecoder):
    """
    Custom Decoder with Ladder model.
    """

    def __init__(
        self,
        opt: Opt,
        embedding: Optional[nn.Embedding] = None,
        n_positions: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(opt, **kwargs)

        def _default(val, default):
            return val if val is not None else default

        opt['dict_size'] = embedding.weight.size(0)
        self.opt = opt
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
        dropout_frac = opt.get('dropout', 0.0)
        self.dropout = nn.Dropout(p=dropout_frac)  # --dropout

        self.n_positions = _default(n_positions, get_n_positions_from_options(opt))
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
                self.position_embeddings.weight, 0, self.embedding_size ** -0.5
            )

        # build the model
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if self.opt['hash_layer'] == i:
                self.layers.append(
                    HashLayer(
                        self.n_heads,
                        self.embedding_size,
                        self.ffn_size,
                        attention_dropout=opt.get('attention_dropout', 0.0),
                        relu_dropout=opt.get('relu_dropout', 0.0),
                        dropout=dropout_frac,
                        activation=self.activation,
                        variant=self.variant,
                        opt=self.opt,
                    )  # type: ignore
                )
            else:
                self.layers.append(
                    self.swappables.layer(
                        self.n_heads,
                        self.embedding_size,
                        self.ffn_size,
                        attention_dropout=opt.get('attention_dropout', 0.0),
                        relu_dropout=opt.get('relu_dropout', 0.0),
                        dropout=dropout_frac,
                        activation=self.activation,
                        variant=self.variant,
                    )  # type: ignore
                )

    def forward_layers(
        self,
        tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        original_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param enc_out:
            encoder outputs
        :param enc_mask:
            encoder output mask
        :param incr_state:
            Dict mapping layer_idx to incremental state

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, encoder_output, encoder_mask, incr_state
            )
        else:
            for _s in range(0, self.opt['ladder_size']):
                for idx, layer in enumerate(self.layers):
                    if idx == self.opt['hash_layer']:
                        tensor, new_incr_state[idx] = layer(
                            x=tensor,
                            encoder_output=encoder_output,
                            encoder_mask=encoder_mask,
                            incr_state=incr_state.get(idx),
                            orig_input=original_input,
                        )
                    else:
                        tensor, new_incr_state[idx] = layer(
                            x=tensor,
                            encoder_output=encoder_output,
                            encoder_mask=encoder_mask,
                            incr_state=incr_state.get(idx),
                        )

        return tensor, new_incr_state

    def forward(self, input, encoder_state, incr_state=None):
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
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, encoder_output, encoder_mask, incr_state, original_input=input
        )

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)

        return tensor, new_incr_state


class HashLayer(TransformerDecoderLayer):
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        ffn_size: int,
        opt: Opt,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__(n_heads, embedding_size, ffn_size, **kwargs)
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = self.swappables.self_attention(
            n_heads, embedding_size, dropout=attention_dropout
        )  # type: ignore
        self.norm1 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = self.swappables.encoder_attention(
            n_heads, embedding_size, dropout=attention_dropout
        )  # type: ignore
        self.norm2 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = HashLayerFFN(
            opt,
            embedding_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=activation,
        )  # type: ignore
        self.norm3 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(
        self, x, encoder_output, encoder_mask, incr_state=None, orig_input=None
    ):
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
            x = _normalize(x, self.norm1)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm1)

        residual = x
        # encoder_attn_layer_norm norm 2
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm2)
        x, final_encoder_attn_incr_state, dotprod = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            incr_state=incr_state.get('encoder_attn'),
            static_kv=True,
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm3)
        x = self.ffn(x, orig_input)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm3)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_encoder_attn_incr_state,
        }

        self.output = x

        return x, new_incr_state


class HashLayerFFN(nn.Module):
    """
    Implements the Hash Layer FFN.
    """

    def __init__(self, opt, dim, dim_hidden, relu_dropout=0, activation='relu'):
        super(HashLayerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.nonlinear = F.relu
        self.opt = opt
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.hashsize = opt['hash_size']

        linears1 = []
        linears2 = []
        norms = []

        for i in range(0, self.hashsize):
            linears1.append(nn.Linear(dim, dim_hidden))
            nn.init.xavier_uniform_(linears1[i].weight)
        for i in range(0, self.hashsize):
            linears2.append(nn.Linear(dim_hidden, dim))
            nn.init.xavier_uniform_(linears2[i].weight)

        embedding_size = self.opt['embedding_size']
        norms.append(LayerNorm(embedding_size, eps=LAYER_NORM_EPS))

        self.linears1 = nn.ModuleList(linears1)
        self.linears2 = nn.ModuleList(linears2)
        self.norms = nn.ModuleList(norms)

        self.alter_tok = -1
        self.alter_bin = -1

    def hash(self, xi):
        # Insert your choice of hash function here.
        # In this code we simply randomly hash based on the given token IDs for simplicity.
        if not hasattr(self, 'hash_bin_map'):
            # create random mapping.
            sz = self.opt['dict_size']
            self.hash_bin_map = torch.LongTensor(sz).fill_(0)
            import random

            random.seed(42)
            for i in range(sz):
                self.hash_bin_map[i] = random.randrange(0, self.hashsize)

        # Now compute the hash bins given the mapping function (Whatever it is).
        return self.hash_bin_map[xi]

    def forward(self, x, orig_input):
        """
        Forward pass.
        """
        xhs = self.hash(orig_input)

        # Now do the real work.
        # This implementation could be more efficient -- but it works.
        index_list = [
            torch.eq(xhs, i).nonzero(as_tuple=True) for i in range(self.hashsize)
        ]
        final_output = x.new_zeros(x.shape)

        for i in range(self.hashsize):
            vecs = x[index_list[i][0], index_list[i][1], :]
            if vecs.shape[0] > 0:
                residual = vecs
                x1 = self.linears1[i](vecs)
                x1 = self.nonlinear(x1)
                x1 = self.relu_dropout(x1)  # --relu-dropout
                x1 = self.linears2[i](x1)
                x1 = residual + x1
                x1 = _normalize(x1, self.norms[0])
                final_output[index_list[i][0], index_list[i][1], :] = x1

        return final_output
