#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Overrides the standard Polyencoder Agent to only return the attention weights.
"""
import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union

from parlai.agents.transformer.modules import (
    MultiHeadAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from parlai.agents.transformer.polyencoder import (
    PolyencoderAgent as BasePolyencoderAgent,
    PolyEncoderModule,
    PolyBasicAttention,
)
from parlai.core.dict import DictionaryAgent
from parlai.core.loader import register_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.utils.torch import PipelineHelper


@register_agent('return_code_weights_agent')
class PolyencoderReturnCodeWeightsAgent(BasePolyencoderAgent):
    """
    A polyencoder agent where the model returns attention weights, rather than encoded
    context.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        BasePolyencoderAgent.add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group('Return Weights Poly Group')
        group.add_argument(
            '--top-k',
            type=int,
            default=100,
            help='How many tokens to output when outputting relevant tokens',
        )
        return parser

    def build_model(self, states=None):
        """
        Return built model.
        """
        return PolyencoderReturnWeightsModule(self.opt, self.dict, self.NULL_IDX)

    def get_ctxt_rep(
        self, batch: Batch, get_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:
        """
        Encode context representation.

        Override to extract weights appropriately.
        """
        ctxt_rep, ctxt_rep_mask, weights, _ = self.model(
            **self._model_context_input(batch)
        )
        return ctxt_rep, ctxt_rep_mask, weights

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Score candidates.

        Override to extract weights appropriately, if needed.
        """
        original_dim = cand_vecs.dim()
        if original_dim == 2:
            cand_vecs = cand_vecs.unsqueeze(1)
        ctxt_rep, ret_weights, ctxt_rep_mask, cand_rep = self.model(
            **self._model_context_input(batch), cand_tokens=cand_vecs
        )
        if original_dim == 2:
            num_cands = cand_rep.size(0)  # will be bsz if using batch cands
            cand_rep = (
                cand_rep.expand(num_cands, batch.text_vec.size(0), -1)
                .transpose(0, 1)
                .contiguous()
            )
        ctxt_code_weights, ctxt_rep_mask = self.model(
            ctxt_rep=ctxt_rep, ctxt_rep_mask=ctxt_rep_mask, cand_rep=cand_rep
        )
        character_weights = torch.bmm(ctxt_code_weights, ret_weights)
        return character_weights, ctxt_rep_mask

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)


class TransformerReturnWeightsEncoderLayer(TransformerEncoderLayer):
    """
    Overridden TransformerEncoderLayer that returns the self-attn weights.
    """

    def forward(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Override to return weights.
        """
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = self.norm1(tensor)
        ####################
        # Begin Difference #
        ####################
        attended_tensor, _, raw_weights, *_ = self.attention(tensor, mask=mask)
        bsz, seq_len, _ = tensor.size()
        weights = (
            raw_weights.view(bsz, self.opt['n_heads'], seq_len, seq_len).max(1).values
        )
        ####################
        # \End Difference  #
        ####################
        tensor = residual + self.dropout(attended_tensor)
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm1(tensor)
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = self.norm2(tensor)
        tensor = residual + self.dropout(self.ffn(tensor))
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm2(tensor)
        tensor *= mask.unsqueeze(-1).type_as(tensor)

        return tensor, weights


class TransformerReturnWeightsEncoder(TransformerEncoder):
    """
    Override TransformerEncoder to return the self-attn weights.
    """

    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, torch.BoolTensor, Optional[torch.Tensor]],
    ]:
        """
        Forward pass.

        Propagate kwargs
        """
        # embed input
        tensor, mask = self.forward_embedding(input, positions, segments)

        if self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)

        # apply transformer layers
        tensor = self.forward_layers(tensor, mask, **kwargs)

        ###################
        # BEGIN DIFFERENCE#
        ###################
        tensor, weights = tensor
        ###################
        # \End  DIFFERENCE#
        ###################

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        # reduce output
        tensor, out_mask = self.reduce_output(tensor, mask)
        if out_mask is not None:
            return tensor, out_mask, weights
        else:
            return tensor, weights

    def forward_layers(
        self, tensor: torch.Tensor, mask: torch.BoolTensor, **kwargs
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        Override to return attention weights.
        """
        weights = None
        if getattr(self.layers, 'is_model_parallel', False):
            # factored out for readability. It is equivalent to the other
            # condition
            tensor, weights = self._apply_model_parallel(tensor, mask, **kwargs)
        else:
            for i in range(self.n_layers):
                tensor, weights = self.layers[i](tensor, mask, **kwargs)
        return tensor, weights

    def _apply_model_parallel(
        self, tensor, mask, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override to return attention weights.
        """
        chunks = PipelineHelper.split((tensor, mask))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        for chunk_idx, layer_nos, next_device in work_items:
            s_weights = None
            try:
                s_tensor, s_mask = chunks[chunk_idx]
            except ValueError:
                s_tensor, s_mask, s_weights = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor, s_weights = self.layers[layer_no](s_tensor, s_mask, **kwargs)
            chunks[chunk_idx] = PipelineHelper.chunk_to(
                (s_tensor, s_mask, s_weights), next_device
            )
        joined = PipelineHelper.join(chunks)
        tensor_out, out_mask, weights = joined
        return tensor_out, weights


class PolyencoderReturnWeightsModule(PolyEncoderModule):
    """
    Constructs attentions and saves their weights!
    """

    def __init__(self, opt: Opt, dict_: DictionaryAgent, null_idx: int):
        super().__init__(opt, dict_, null_idx)
        self.opt = opt
        assert self.type == 'codes'
        if isinstance(self.code_attention, PolyBasicAttention):
            self.code_attention.get_weights = True
        if self.attention_type != 'multihead':
            self.attention.get_weights = True

    def get_encoder(self, opt, dict_, null_idx, reduction_type, for_context: bool):
        """
        Override to not build the cand encoder.
        """
        if not for_context:
            wrapped_class = TransformerEncoder
        else:
            wrapped_class = TransformerReturnWeightsEncoder.with_components(
                layer=TransformerReturnWeightsEncoderLayer
            )
        embeddings = self._get_embeddings(
            dict_=dict_, null_idx=null_idx, embedding_size=opt['embedding_size']
        )
        return wrapped_class(
            opt=opt,
            embedding=embeddings,
            vocabulary_size=len(dict_),
            padding_idx=null_idx,
            reduction_type=reduction_type,
        )

    def attend(
        self,
        attention_layer: torch.nn.Module,
        queries: torch.Tensor,
        keys: Optional[torch.Tensor],
        values: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return attended tensor and weights.
        """
        if keys is None:
            keys = values
        if isinstance(attention_layer, PolyBasicAttention):
            attended, weights = attention_layer(
                queries, keys, mask_ys=mask, values=values
            )
        elif isinstance(attention_layer, MultiHeadAttention):
            attended, _, weights = attention_layer(queries, keys, values, mask)
        else:
            raise Exception('Unrecognized type of attention')

        return attended, weights

    def encode(
        self, cand_tokens: Optional[torch.Tensor], **ctxt_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, Optional[torch.Tensor]]:
        """
        Override Polyencoder.encode to *only* return the coded/self-attn attention
        weights.
        """
        assert len(ctxt_inputs) > 0
        assert 'ctxt_tokens' in ctxt_inputs
        assert len(ctxt_inputs['ctxt_tokens'].shape) == 2
        assert self.type == 'codes'

        cand_embed = None
        if cand_tokens is not None:
            if len(cand_tokens.shape) != 3:
                cand_tokens = cand_tokens.unsqueeze(1)
            bsz = cand_tokens.size(0)
            num_cands = cand_tokens.size(1)
            cand_embed = self.encoder_cand(cand_tokens.view(bsz * num_cands, -1))
            cand_embed = cand_embed.view(bsz, num_cands, -1)

        bsz = self._get_context_batch_size(**ctxt_inputs)
        # get context_representation. Now that depends on the cases.
        ctxt_out, ctxt_mask, ctxt_self_attn_weights = self.encoder_ctxt(
            **self._context_encoder_input(ctxt_inputs)
        )
        ctxt_self_attn_weights = F.softmax(ctxt_self_attn_weights, dim=-1)
        ctxt_rep, ctxt_code_weights = self.attend(
            self.code_attention,
            queries=self.codes.repeat(bsz, 1, 1),
            keys=ctxt_out,
            values=ctxt_out,
            mask=ctxt_mask,
        )

        return ctxt_rep, ctxt_code_weights, ctxt_mask, cand_embed

    def score(
        self,
        ctxt_rep: torch.Tensor,
        ctxt_rep_mask: torch.Tensor,
        cand_embed: torch.Tensor,
    ):
        """
        Override score to return the attention weights **RATHER THAN THE SCORES**
        """
        ones_mask = ctxt_rep.new_ones(
            ctxt_rep.size(0), self.n_codes
        ).byte()  # type: ignore
        ctxt_final_rep, ctxt_code_weights = self.attend(
            self.attention, cand_embed, ctxt_rep, ctxt_rep, ones_mask  # type: ignore
        )
        return ctxt_code_weights, ctxt_rep_mask
