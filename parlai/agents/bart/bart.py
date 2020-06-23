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

from parlai.agents.bart.modules import BartModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import compare_init_model_opts
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch, History
from parlai.scripts.convert_fairseq_to_parlai import ConversionScript
from parlai.utils.typing import TShared
from parlai.zoo.bart.build import download, CONVERSION_ARGS, BART_ARGS


class BartAgent(TransformerGeneratorAgent):
    """
    BART Agent.

    Relies on BART model implemented in fairseq.

    Can initialize with BART, or can handle
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        if not shared:
            if not opt.get('converting'):
                download(opt['datapath'])
                opt['init_model'] = os.path.join(opt['datapath'], 'models/bart_models/bart.large/model')
            if opt.get('init_fairseq_model'):
                opt = self._convert_model(opt)
            opt.update(BART_ARGS)
            compare_init_model_opts(opt, opt)
        super().__init__(opt, shared)

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = BartModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def _get_initial_decoder_input(self, bsz, beam_size, dev):
        """
        Override to seed decoder with EOS token.
        """
        return (
            torch.LongTensor([self.END_IDX]).expand(bsz * beam_size, 1).to(dev)
        )

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None
    ):
        """
        Override to set prefix_tokens.

        For bart pretraining, a bos token was added to the input.
        so basically the input to encoder is:

        <bos> seq <eos>

        input to decoder:
        <eos> <bos> seq

        target is:
        <bos> seq <eos>
        """
        if batch.text_vec is not None:
            prefix_tokens = batch.text_vec.new_zeros((len(batch.text_vec), 1)).fill_(self.START_IDX)
        return super()._generate(batch, beam_size, max_ts, prefix_tokens)

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
        if opt.get('model_file') and not os.path.exists(opt['model_file']):
            args['output'] = opt['model_file']
        else:
            args['output'] = os.path.join(opt['datapath'], 'models/converted_fairseq_models/', model_name)
        ConversionScript.main(**args)
        opt['init_model'] = args['output']
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
        Override to prepend start token and append end token.
        """
        obs = super()._set_text_vec(obs, history, truncate)
        if 'text' not in obs or 'text_vec' not in obs:
            return obs
        vec = obs['text_vec']
        if truncate is not None:
            vec = torch.LongTensor(
                self._check_truncate(obs['text_vec'], truncate - 2, True)
            )
        obs.force_set(
            'text_vec',
            torch.cat(
                [
                    vec.new_tensor([self.dict[self.dict.start_token]]),
                    vec,
                    vec.new_tensor([self.dict[self.dict.end_token]])],
                0
            )
        )
        return obs
