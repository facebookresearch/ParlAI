#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.agents.hugging_face.dict import DialoGPTDictionaryAgent
from parlai.agents.hugging_face.gpt2 import Gpt2Agent, GPT2Decoder, HFGPT2Model
from parlai.utils.misc import warn_once

try:
    from transformers import GPT2Model
except ImportError:
    raise ImportError('Please run `pip install transformers`.')


############################################
## Modules
############################################


class DialoGPTDecoder(GPT2Decoder):
    """
    DialoGPT Decoder.

    This decoder is initialized with the pretrained model from Hugging Face.
    """

    def _init_from_pretrained(self, opt):
        # load model
        model_sz = opt['gpt2_size']
        fle_key = f'microsoft/DialoGPT-{model_sz}'
        return GPT2Model.from_pretrained(fle_key)


class DialoGPTModel(HFGPT2Model):
    """
    Hugging Face DialoGPT Model.
    """

    def _get_decoder(self, opt, dict):
        return DialoGPTDecoder(opt, dict)


############################################
## Agent
############################################


class DialogptAgent(Gpt2Agent):
    """
    Hugging Face DialoGPT Agent.

    DialoGPT is based on GPT2, which is a multi-layer decoder-only Transformer.
    The decoder is initialized with pretrained weights from Hugging Face.
    Read more about this model here
    <https://huggingface.co/transformers/model_doc/dialogpt.html>.

    DialoGPT comes in three  sizes: small, medium, large.

    If you are finetuning the Dialogpt Agent as a dialogue agent, be sure
    to run `--add-special-tokens True`. To examine the performance of the
    agent out of the box, run with `--add-special-tokens False`, and make
    sure that the batch size is 1.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('DialoGPT Args')
        agent.add_argument(
            '--gpt2-size',
            type=str,
            default='small',
            choices=['small', 'medium', 'large'],
            help='Which size model to initialize.',
        )
        argparser.set_defaults(
            delimiter='<|endoftext|>',
            history_add_global_end_token='<|endoftext|>',
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super(DialogptAgent, cls).add_cmdline_args(argparser)
        warn_once('WARNING: this model is in beta and the API is subject to change.')
        return agent

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return DialoGPTDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return DialoGPTModel(self.opt, self.dict)
