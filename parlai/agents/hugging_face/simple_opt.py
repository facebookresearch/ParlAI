#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import os
from typing import Optional, Dict, Any, Tuple

import torch

# from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent, OPTDictionaryAgent
from parlai.agents.hugging_face.dict import OPTDictionaryAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, padded_tensor

try:
    # from transformers import GPT2Model, AutoModelForCausalLM
    from transformers import AutoModelForCausalLM
except ImportError:
    raise ImportError("Please run `pip install transformers`.")


def set_device(func):
    """
    Decorator for setting device.

    HF's model parallel uses `torch.cuda.set_device`, which does not vibe well with
    ParlAI.
    """

    def wrap(*args, **kwargs):
        self = args[0]
        # self.paralleled implies whether the model has been paralleled.
        # it is set to the opposite of `opt['t5_model_parallel]`
        parallel = hasattr(self, 'paralleled') and not self.paralleled
        if torch.cuda.is_available() and parallel:
            torch.cuda.set_device('cuda:0')
        ret = func(*args, **kwargs)
        if torch.cuda.is_available() and parallel:
            torch.cuda.set_device('cuda:0')
        return ret

    return wrap


############################################
## Modules
############################################


class ParlaiOPTDecoder(torch.nn.Module):
    def __init__(self, opt: Opt, dict, decoder, padding_idx: Optional[int] = None):
        super().__init__()
        # self.stack = decoder
        # self.opt_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
        self.opt_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        self.padding_idx = padding_idx
        self.add_start_token = False
        self.START_IDX = dict.start_idx
        self.NULL_IDX = dict.null_idx
        self.END_IDX = dict.end_idx
        # self.paralleled = not opt[
        #     't5_model_parallel'
        # ]  # need to parallel in forward; bug in HF

    def forward(self, input, encoder_state, incr_state=None):
        attention_mask = None
        position_ids = None
        if incr_state is None:
            # first step
            if (
                not self.add_start_token
                and input.size(1) == 1
                and int(input[0][0]) == self.START_IDX
            ):
                # generating: ignore the start token
                # without deep copy, the padding_idx (-1) in encoder_state can be reset to 0 with clamp_ inplace operation
                model_input = encoder_state.clone()
            else:
                # forced decoding: concatenate the context
                # with the labels
                model_input = torch.cat([encoder_state, input], dim=-1)
            attention_mask = model_input != self.NULL_IDX
            position_ids = (
                attention_mask.cumsum(dim=-1, dtype=torch.int64) - 1
            ).clamp_(min=0)
        else:
            if not self.add_start_token:
                input = input[:, 1:]
            # generating with continuation
            # get the position ids
            position_ids = (encoder_state != self.NULL_IDX).sum(
                -1, True, dtype=torch.int64
            ) - 1
            delta = ((input != self.NULL_IDX)).sum(-1, True, dtype=torch.int64)
            position_ids += delta
            # generation: get the last token input
            model_input = input[:, -1:]
            attention_mask = torch.cat([encoder_state, input], dim=-1) != self.NULL_IDX

        model_input = model_input.clamp_(min=0)
        transformer_outputs = self.opt_model(
            model_input,
            past_key_values=incr_state,
            attention_mask=attention_mask,
            # position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        if incr_state is None:
            # pull out only the hidden states for the label tokens
            output = hidden_states[:, -input.size(1) - 1 + int(self.add_start_token) :]
            # hack: we need the last state of the encoder-side to be the first
            # element of the decoder-side
            lengths = (input != self.NULL_IDX).sum(dim=-1)
            for i in range(input.size(0)):
                output[i, input.size(1) - lengths[i]] = output[i, 0]

        else:
            # generation, we're only doing one token at a time. no need to
            # shove things back in
            output = hidden_states

        return output, new_incr_state

    # @set_device
    # def forward(
    #     self, input: torch.LongTensor, encoder_state: Tuple[Any], incr_state=None
    # ):
    #     """
    #     Forward pass.

    #     :param LongTensor[batch,seqlen] input:
    #         The decoder inputs (partial or full decoded token IDs).
    #     :param encoder_state:
    #         Output from the encoder module forward pass.
    #     :param incr_state:
    #         The incremental state: a dictionary whose keys index the layers and whose
    #         values contain the incremental state for each layer.
    #     """
    #     # if not self.paralleled:
    #     #     self.stack.parallelize()
    #     print(encoder_state)
    #     print(input)
    #     encoder_output, encoder_mask = encoder_state

    #     mask = input != self.padding_idx
    #     mask[:, 0] = True  # first token is pad

    #     outputs = self.stack(
    #         input_ids=input,
    #         attention_mask=mask,
    #         encoder_hidden_states=encoder_output.to(input.device),
    #         encoder_attention_mask=encoder_mask.to(input.device),
    #     )
    #     return outputs[0].to(input.device), incr_state


class HFOPTModel(TorchGeneratorModel):
    """
    Hugging Face GPT2 Model.

    GPT2 is a multi-layer decoder-only Transformer. As such, the encoder
    is simply an identity layer. The decoder is initialized with pretrained
    weights from Hugging Face. Read more about this model here
    <https://huggingface.co/transformers/model_doc/gpt2.html>.
    """

    def __init__(self, opt, dict):
        self.add_start_token = opt["add_start_token"]
        super().__init__(*self._get_special_tokens(opt, dict))
        self.opt_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        # init the model
        # self.encoder = ParlaiT5Encoder(opt, self.opt_model.get_encoder(), self.pad_idx)
        self.encoder = IdentityLayer()
        self.pad_idx = dict[dict.null_token]
        self.decoder = ParlaiOPTDecoder(
            opt, dict, self.opt_model.get_decoder(), self.pad_idx
        )
        # self.encoder = IdentityLayer()
        # self.decoder = self.opt_model.get_decoder()
        # self.decoder = self._get_decoder(opt, dict)
        # self.config = self.decoder.transformer.config
        # self.lm_head = torch.nn.Linear(
        #     self.config.n_embd, self.config.vocab_size, bias=False
        # )
        self.lm_head = self.opt_model.lm_head
        # add start token

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        enc = torch.index_select(encoder_states, 0, indices)
        return enc

    def output(self, tensor):
        """
        Compute output logits.
        """
        return tensor
        # return self.lm_head(tensor)

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = []
        for layer_past in incremental_state:
            if torch.is_tensor(layer_past):
                new_incr_state.append(torch.index_select(layer_past, 1, inds))
            else:
                # newer versions of HF split up the intermediate outputs
                assert isinstance(layer_past, tuple)
                layer_past = torch.stack(layer_past, dim=0)
                new_incr_state.append(torch.index_select(layer_past, 1, inds))

        return tuple(new_incr_state)

    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        if self.add_start_token:
            return super().decode_forced(encoder_states, ys)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, _ = self.decoder(inputs, encoder_states)
        # latent, _ = self.opt_model.get_decoder()(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds


############################################
## Agent
############################################


class SimpleOptAgent(TorchGeneratorAgent):
    """
    Hugging Face GPT2 Agent.

    GPT2 is a multi-layer decoder-only Transformer.
    The decoder is initialized with pretrained weights from Hugging Face.
    Read more about this model here
    <https://huggingface.co/transformers/model_doc/gpt2.html>.

    GPT2 comes in five sizes: distilgpt2, small, medium, large, XL. Use the
    flag `--gpt2-size` to choose the size.

    If you are finetuning the Gpt2 agent as a dialogue agent, be sure
    to run `--add-special-tokens True`. To examine the performance of the
    agent out of the box, run with `--add-special-tokens False`, and make
    sure that the batch size is 1.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group("OPT Args")
        agent.add_argument(
            "--add-special-tokens",
            type="bool",
            default=True,
            help="Add special tokens (like PAD, etc.). If False, "
            "Can only use with batch size 1.",
        )
        agent.add_argument(
            "--add-start-token",
            type="bool",
            default=False,
            help="Add start tokens when finetuning.",
        )
        parser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        warn_once("WARNING: this model is in beta and the API is subject to change.")
        return agent

    def __init__(self, opt, shared=None):
        if not opt["add_special_tokens"] and opt.get('batchsize', 1) > 1:
            # *** STOP ***
            # You may be a future researcher who has stumbled upon this odd
            # restriction, and is tempted to comment this out. After all, the
            # code still runs when it's uncommented, why shouldn't you?
            # You should know this has serious implications, as gpt2 doesn't have
            # padding tokens. This is incompatible with ParlAI's batching,
            # which puts conversations of different length in the same
            # batch. Without a padding token, nonsense will be inserted into
            # the context, and the generations & PPL you get will be wrong.
            raise RuntimeError(
                "If using batchsize > 1, --add-special-tokens must be True."
            )
        if not opt["add_special_tokens"] and opt["add_start_token"]:
            raise RuntimeError(
                "--add-start-token true requires --add-special-tokens true"
            )
        super().__init__(opt, shared)
        if hasattr(self.model, "module"):
            self.START_IDX = self.model.module.START_IDX
            self.END_IDX = self.model.module.END_IDX
            self.NULL_IDX = self.model.module.NULL_IDX
        else:
            self.START_IDX = self.model.START_IDX
            self.END_IDX = self.model.END_IDX
            self.NULL_IDX = self.model.NULL_IDX

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overridden if a more complex dictionary is required.
        """
        return OPTDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return HFOPTModel(self.opt, self.dict)

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    def _pad_tensor(self, items, is_label=False):
        """
        Override to always set fp16friendly to False and left_pad to True.
        """
        return padded_tensor(
            items, pad_idx=self.NULL_IDX, left_padded=True, fp16friendly=False
        )

    def load_state_dict(self, state_dict):
        # 2020-11-10: some very old transformer model points (pre v3.0.1) are
        # missing a field called transformer.h.0.attn.masked_bias. This hacks
        # around that. See
        # https://github.com/huggingface/transformers/issues/4309.
        current_sd = self.model.state_dict()
        missing = set(current_sd.keys()) - set(state_dict.keys())
        for m in missing:
            if 'masked_bias' in m:
                state_dict[m] = current_sd[m]
        return super().load_state_dict(state_dict)
