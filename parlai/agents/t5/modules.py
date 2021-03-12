#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Wrapped Encoders for ParlAI Use
"""
import torch
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from typing import Optional, Dict, Any, Tuple

from parlai.core.opt import Opt
from parlai.core.torch_generator_agent import TorchGeneratorModel

from parlai.agents.t5.dict import T5TokenizerDictionaryAgent


def build_t5(opt: Opt) -> T5ForConditionalGeneration:
    return T5ForConditionalGeneration.from_pretrained(
        opt['t5_model_arch'], dropout_rate=opt['t5_dropout']
    )


def set_device(func):
    """
    Decorator for setting device.

    HF's model parallel uses `torch.cuda.set_device`, which does not
    vibe well with ParlAI.
    """

    def wrap(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
        ret = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
        return ret

    return wrap


class ParlaiT5Encoder(torch.nn.Module):
    def __init__(
        self, opt: Opt, encoder: T5Stack, dictionary: T5TokenizerDictionaryAgent
    ):
        super().__init__()
        self.stack = encoder
        self.padding_idx = dictionary[dictionary.null_token]
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        if not self.paralleled:
            self.stack.parallelize()
        mask = input != self.padding_idx
        outputs = self.stack(input, attention_mask=mask, output_hidden_states=False)
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        return outputs[0], mask


class ParlaiT5Decoder(torch.nn.Module):
    def __init__(
        self, opt: Opt, decoder: T5Stack, dictionary: T5TokenizerDictionaryAgent
    ):
        super().__init__()
        self.stack = decoder
        self.padding_idx = dictionary[dictionary.null_token]
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self, input: torch.LongTensor, encoder_state: Tuple[Any], incr_state=None
    ):
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
        if not self.paralleled:
            self.stack.parallelize()
        encoder_output, encoder_mask = encoder_state

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
        else:
            incr_state = {}
        mask = input != self.padding_idx
        mask[:, 0] = True

        outputs = self.stack(
            input_ids=input,
            attention_mask=mask,
            encoder_hidden_states=encoder_output.to(input.device),
            encoder_attention_mask=encoder_mask.to(input.device),
        )
        return outputs[0].to(input.device), incr_state


class ParlaiT5Model(TorchGeneratorModel):
    """
    Wrap T5 in ParlAI.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.t5 = build_t5(opt)
        self.encoder = ParlaiT5Encoder(opt, self.t5.get_encoder(), dictionary)
        self.decoder = ParlaiT5Decoder(opt, self.t5.get_decoder(), dictionary)

    @set_device
    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
        return inputs

    @set_device
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
        Irrelevant as HF generates for us.
        """
        return {}

    @set_device
    def output(self, tensor):
        """
        Compute output logits.
        """
        # Taken directly from HuggingFace
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        tensor = tensor * (self.t5.model_dim ** -0.5)
        lm_logits = self.t5.lm_head(tensor)
        return lm_logits
