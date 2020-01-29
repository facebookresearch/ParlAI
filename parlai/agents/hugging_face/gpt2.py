#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
from parlai.utils.misc import warn_once

from transformers import GPT2Model

import torch

############################################
## Modules
############################################


class DummyEncoder(torch.nn.Module):
    """
    Dummy encoder that just does a forward pass here.
    """

    def forward(self, xs):
        """
        Identity.
        """
        return xs


class GPT2Decoder(torch.nn.Module):
    def __init__(self, opt, dict):
        super().__init__()
        model_sz = opt['gpt2_size']
        fle_key = 'gpt2' if model_sz == 'small' else f'gpt2-{model_sz}'
        self.start_idx = dict.start_idx
        self.pad_idx = dict.pad_idx
        self.transformer = GPT2Model.from_pretrained(fle_key)
        self.transformer.resize_token_embeddings(len(dict.tokenizer))

    def forward(self, input, encoder_state, incr_state=None):
        attention_mask = None
        if incr_state is None:
            # first step
            if input.size(1) == 1 and int(input[0][0]) == self.start_idx:
                # generating: ignore the start token
                model_input = encoder_state
            else:
                # forced decoding: concatenate the context
                # with the labels
                model_input = torch.cat([encoder_state, input], 1)
                attention_mask = model_input != self.pad_idx
        else:
            # generation: get the last token input
            model_input = input[:, -1].unsqueeze(1)

        transformer_outputs = self.transformer(
            model_input, past=incr_state, attention_mask=attention_mask
        )
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        return hidden_states, new_incr_state


class HFGPT2Model(TorchGeneratorModel):
    def __init__(self, opt, dict):
        self.pad_idx, self.start_idx, self.end_idx = self._get_special_tokens(opt, dict)
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)

        # init the model
        self.encoder = DummyEncoder()
        self.decoder = GPT2Decoder(opt, dict)

        self.config = self.decoder.transformer.config
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )
        self.tie_weights(self.lm_head, self.decoder.transformer.wte)

        self.enc_sz = None

    def tie_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def _get_special_tokens(self, opt, dict):
        return dict.pad_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        enc = torch.index_select(encoder_states, 0, indices)
        return enc

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        if self.enc_sz is not None and self.enc_sz[1] - 1 < tensor.size(1):
            # keep only label scores
            # -1 here is because we do not at a start token.
            tensor = tensor[:, self.enc_sz[1] - 1 :, :]
        return self.lm_head(tensor)

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = []
        for layer_past in incremental_state:
            reordered_layer_past = [
                layer_past[:, i].unsqueeze(1).clone().detach() for i in inds
            ]
            reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
            new_incr_state.append(reordered_layer_past)

        return tuple(new_incr_state)

    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        if ys is not None:
            self.enc_sz = xs[0].size()
        else:
            self.enc_sz = None

        return super().forward(*xs, ys=ys, prev_enc=prev_enc, maxlen=maxlen, bsz=bsz)


############################################
## Agent
############################################


class Gpt2Agent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Gpt2 Args')
        agent.add_argument(
            '--gpt2-size',
            type=str,
            default='small',
            choices=['small', 'medium', 'large', 'xl'],
            help='Which size model to initialize.',
        )
        argparser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super(Gpt2Agent, cls).add_cmdline_args(argparser)
        warn_once('WARNING: this model is in beta and the API is ' 'subject to change.')
        return agent

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
