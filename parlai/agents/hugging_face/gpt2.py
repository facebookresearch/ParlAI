#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.torch import DummyEncoder, concat_without_padding

from transformers import GPT2Model

import torch

############################################
## Modules
############################################


class GPT2Decoder(torch.nn.Module):
    def __init__(self, opt, dict):
        super().__init__()
        # load model
        model_sz = opt['gpt2_size']
        fle_key = 'gpt2' if model_sz == 'small' else f'gpt2-{model_sz}'
        self.transformer = GPT2Model.from_pretrained(fle_key)
        # add special tokens
        self.start_idx = dict.start_idx
        self.null_idx = dict.null_idx
        if opt['add_special_tokens']:
            self.transformer.resize_token_embeddings(len(dict.tokenizer))
        # use cuda
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()

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
                model_input, _ = concat_without_padding(
                    encoder_state,
                    input,
                    use_cuda=self.use_cuda,
                    null_idx=self.null_idx,
                )
                attention_mask = model_input != self.null_idx
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
        self.null_idx, self.start_idx, self.end_idx = self._get_special_tokens(
            opt, dict
        )
        super().__init__(self.null_idx, self.start_idx, self.end_idx)

        # init the model
        self.encoder = DummyEncoder()
        self.decoder = GPT2Decoder(opt, dict)

        self.config = self.decoder.transformer.config
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )
        self.tie_weights(self.lm_head, self.decoder.transformer.wte)

        self.text_lengths = None  # used to reverse concatenation of context and labels

    def tie_weights(self, output_embeddings, input_embeddings):
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
        # get only scores for labels
        if self.text_lengths is not None:
            total_length = max(self.text_lengths)
            to_select = tensor.size(1) - total_length + 1
            if to_select > 0:
                # select only label scores
                bsz = tensor.size(0)
                new_tensors = []
                for i in range(bsz):
                    start = self.text_lengths[i] - 1
                    end = start + to_select
                    new_tensors.append(tensor[i, start:end, :].unsqueeze(0))
                tensor = torch.cat(new_tensors, 0)

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
        model_input, text_lengths = xs
        if ys is not None:
            self.text_lengths = text_lengths
        else:
            self.text_lengths = None

        return super().forward(
            model_input, ys=ys, prev_enc=prev_enc, maxlen=maxlen, bsz=bsz
        )


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
        agent.add_argument(
            '--add-special-tokens',
            type='bool',
            default=True,
            help='Add special tokens (like PAD, etc.). If False, '
            'Can only use with batch size 1.',
        )
        argparser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super(Gpt2Agent, cls).add_cmdline_args(argparser)
        warn_once('WARNING: this model is in beta and the API is ' 'subject to change.')
        return agent

    def __init__(self, opt, shared=None):
        if not opt['add_special_tokens'] and opt['batchsize'] > 1:
            raise RuntimeError(
                'If using batchsize > 1, --add-special-tokens ' 'must be True.'
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

    def _model_input(self, batch):
        """
        Override to pass in text lengths.
        """
        return (batch.text_vec, batch.text_lengths)

    def _encoder_input(self, batch):
        return (batch.text_vec,)
