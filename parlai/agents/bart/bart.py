#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension

See https://arxiv.org/abs/1910.13461.
"""
import os
import torch
from typing import Optional

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import compare_init_model_opts
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.torch_agent import History
from parlai.scripts.convert_fairseq_to_parlai import ConversionScript
from parlai.utils.typing import TShared
from parlai.zoo.bart.build import download, CONVERSION_ARGS


class BartAgent(TransformerGeneratorAgent):
    """
    BART Agent.

    Relies on BART model implemented in fairseq.

    Can initialize with BART, or can handle
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        if not opt.get('converting'):
            download(opt['datapath'])
            opt['init_model'] = os.path.join(opt['datapath'], 'models/bart_models/bart.large/model')
        if opt.get('init_fairseq_model'):
            opt = self._convert_model(opt)
        opt = self._set_bart_args(opt)
        compare_init_model_opts(opt, opt)
        super().__init__(opt, shared)

    def _convert_model(self, opt: Opt) -> Opt:
        """
        Convert fairseq init model to ParlAI Model.

        :param opt:
            options

        :return opt:
            return opt with new init_model path
        """
        model_name = os.path.split(opt['init_fairseq_model'])[-1]
        args = CONVERSION_ARGS.copy()
        args['input'] = [opt['init_fairseq_model']]
        args['output'] = os.path.join(opt['datapath'], 'models/bart_models/converted_models/', model_name)
        ConversionScript.main(**args)
        opt['init_model'] = args['output']
        return opt

    def _set_bart_args(self, opt: Opt) -> Opt:
        """
        Set BART-specific args.

        :param opt:
            opt

        :return opt:
            return new opt with BART args.
        """
        opt.update(
            {
                'embedding_size': 1024,
                'ffn_size': 4096,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'n_heads': 16,
                'n_positions': 1024,
                'variant': 'xlm',
                'activation': 'gelu',
                'n_encoder_layers': 12,
                'n_decoder_layers': 12,
                'force_fp16_tokens': True,
                'history_add_global_end_token': True,
                'dict_tokenizer': 'gpt2',
                'embeddings_scale': False
            }
        )
        return opt

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Override to add one arg.
        """
        TransformerGeneratorAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Bart Args')
        group.add_argument(
            '--init-fairseq-model',
            type=str,
            default=None,
            help='fairseq checkpoint for bart',
        )

    def _set_text_vec(
        self,
        obs: Message,
        history: History,
        truncate: Optional[int],
    ) -> Message:
        """
        Override to prepend start token.
        """
        obs = super()._set_text_vec(obs, history, truncate)
        if 'text' not in obs or 'text_vec' not in obs:
            return obs
        vec = obs['text_vec']
        if truncate is not None:
            vec = torch.LongTensor(
                self._check_truncate(obs['text_vec'], truncate - 1, True)
            )
        obs.force_set(
            'text_vec',
            torch.cat([vec.new_tensor([self.dict[self.dict.start_token]]), vec], 0)
        )
        return obs
