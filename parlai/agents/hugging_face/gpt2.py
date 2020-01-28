#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent

from transformers import GPT2Model

import torch


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
        self.transformer = GPT2Model.from_pretrained(fle_key)

    def forward(self, input, encoder_state, incr_state=None):
        if incr_state is None:
            # we are on the first step
            model_input = encoder_state
        else:
            # get rid of START token
            # TODO: consider whether we always want to do this
            model_input = input[:, -1].unsqueeze(1)
        transformer_outputs = self.transformer(model_input, past=incr_state,)
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
        super(Gpt2Agent, cls).add_cmdline_args(argparser)
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
