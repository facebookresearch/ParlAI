#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor


try:
    from transformers import GPT2Model
except ImportError:
    raise ImportError("Please run `pip install transformers`.")


############################################
## Modules
############################################


class GPT2Decoder(torch.nn.Module):
    """
    GPT2 Decoder.

    This decoder is initialized with the pretrained model from Hugging Face.
    """

    def __init__(self, opt, dict):
        super().__init__()
        self.transformer = self._init_from_pretrained(opt)
        # add special tokens
        self.start_idx = dict.start_idx
        self.null_idx = dict.null_idx
        self.add_start_token = False
        if opt["add_special_tokens"]:
            self.transformer.resize_token_embeddings(len(dict.tokenizer))
            self.add_start_token = opt["add_start_token"]
        # use cuda
        self.use_cuda = not opt["no_cuda"] and torch.cuda.is_available()

    def _init_from_pretrained(self, opt):
        # load model
        # check if datapath has the files that hugging face code looks for
        if all(
            os.path.isfile(
                os.path.join(opt["datapath"], "models", "gpt2_hf", file_name)
            )
            for file_name in ["pytorch_model.bin", "config.json"]
        ):
            fle_key = os.path.join(opt["datapath"], "models", "gpt2_hf")
        else:
            model_sz = opt["gpt2_size"]
            if model_sz == "small":
                fle_key = "gpt2"
            elif model_sz == "distilgpt2":
                fle_key = "distilgpt2"
            else:
                fle_key = f"gpt2-{model_sz}"
        return GPT2Model.from_pretrained(fle_key)

    def forward(self, input, encoder_state, incr_state=None):
        # __import__("ipdb").set_trace()  # FIXME
        attention_mask = None
        position_ids = None
        if incr_state is None:
            # first step
            if (
                not self.add_start_token
                and input.size(1) == 1
                and int(input[0][0]) == self.start_idx
            ):
                # generating: ignore the start token
                model_input = encoder_state
            else:
                # forced decoding: concatenate the context
                # with the labels
                model_input = torch.cat([encoder_state, input], dim=-1)
            attention_mask = model_input != self.null_idx
            position_ids = (
                attention_mask.cumsum(dim=-1, dtype=torch.int64) - 1
            ).clamp_(min=0)
        else:
            if not self.add_start_token:
                input = input[:, 1:]
            # generating with continuation
            # get the position ids
            position_ids = (encoder_state != self.null_idx).sum(
                -1, True, dtype=torch.int64
            ) - 1
            delta = ((input != self.null_idx)).sum(-1, True, dtype=torch.int64)
            position_ids += delta
            # generation: get the last token input
            model_input = input[:, -1:]
            attention_mask = torch.cat([encoder_state, input], dim=-1) != self.null_idx

        transformer_outputs = self.transformer(
            model_input,
            past=incr_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        if incr_state is None:
            # pull out only the hidden states for the label tokens
            output = hidden_states[:, -(input.size(1) + 1) :]
            # hack: we need the last state of the encoder-side to be the first
            # element of the decoder-side
            lengths = (input != self.null_idx).sum(dim=-1)
            for i in range(input.size(0)):
                output[i, input.size(1) - lengths[i]] = output[i, 0]

        else:
            # generation, we're only doing one token at a time. no need to
            # shove things back in
            output = hidden_states

        return output, new_incr_state


class HFGPT2Model(TorchGeneratorModel):
    """
    Hugging Face GPT2 Model.

    GPT2 is a multi-layer decoder-only Transformer. As such, the encoder
    is simply an identity layer. The decoder is initialized with pretrained
    weights from Hugging Face. Read more about this model here
    <https://huggingface.co/transformers/model_doc/gpt2.html>.
    """

    def __init__(self, opt, dict):
        self.null_idx, self.start_idx, self.end_idx = self._get_special_tokens(
            opt, dict
        )
        super().__init__(self.null_idx, self.start_idx, self.end_idx)

        # init the model
        self.encoder = IdentityLayer()
        self.decoder = self._get_decoder(opt, dict)
        self.config = self.decoder.transformer.config
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )
        self._tie_weights(self.lm_head, self.decoder.transformer.wte)
        # add start token
        self.add_start_token = opt["add_special_tokens"] and opt["add_start_token"]

    def _get_decoder(self, opt, dict):
        return GPT2Decoder(opt, dict)

    def _tie_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        enc = torch.index_select(encoder_states, 0, indices)
        return enc

    def output(self, tensor):
        """
        Compute output logits.
        """
        return self.lm_head(tensor)

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = []
        for layer_past in incremental_state:
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
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds


############################################
## Agent
############################################


class Gpt2Agent(TorchGeneratorAgent):
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
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group("Gpt2 Args")
        agent.add_argument(
            "--gpt2-size",
            type=str,
            default="small",
            choices=["small", "medium", "large", "xl", "distilgpt2"],
            help="Which size model to initialize.",
        )
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
        argparser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super(Gpt2Agent, cls).add_cmdline_args(argparser)
        warn_once("WARNING: this model is in beta and the API is subject to change.")
        return agent

    def __init__(self, opt, shared=None):
        if not opt["add_special_tokens"] and opt["batchsize"] > 1:
            raise RuntimeError(
                "If using batchsize > 1, --add-special-tokens must be True."
            )
        super().__init__(opt, shared)

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return Gpt2DictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return HFGPT2Model(self.opt, self.dict)

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    def _pad_tensor(self, items):
        """
        Override to always set fp16friendly to False and left_pad to True.
        """
        return padded_tensor(
            items,
            pad_idx=self.NULL_IDX,
            use_cuda=self.use_cuda,
            left_padded=True,
            fp16friendly=False,
        )
