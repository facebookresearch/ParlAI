#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

See <https://arxiv.org/abs/1910.10683>

The T5 agent can be instantiated as simply `-m t5`
"""
import torch
from typing import Optional, Dict, Any

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from parlai.agents.t5.dict import T5TokenizerDictionaryAgent
from parlai.agents.t5.modules import ParlaiT5Model
from parlai.agents.t5.task_specific_configs import TASK_CONFIGS


class T5Agent(TorchGeneratorAgent):
    """
    T5 Agent.

    Relies on the T5 model implemented in huggingface
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('T5 Args')
        group.add_argument(
            '--t5-model-arch',
            type=str,
            default='t5-base',
            choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
        )
        group.add_argument(
            '--t5-model-parallel',
            type='bool',
            default=False,
            help='use HF model parallel',
        )
        group.add_argument(
            '--t5-dropout', type=float, default=0.0, help='Dropout for T5'
        )
        group.add_argument(
            '--t5-generation-config',
            type=str,
            default=None,
            choices=[
                'summarization',
                'translation_en_to_de',
                'translation_en_to_fr',
                'translation_en_to_ro',
            ],
            help='Task specific generation config for T5',
        )
        return parser

    def build_model(self) -> ParlaiT5Model:
        """
        Build and return model.
        """
        model = ParlaiT5Model(self.opt, self.dict)
        if self.opt['t5_model_parallel']:
            model.t5.parallelize()
        return model

    def build_dictionary(self):
        """
        Overrides TorchAgent.build_dictionary to use t5 dict.
        """
        return T5TokenizerDictionaryAgent(self.opt)

    def observe(self, observation):
        """
        Override to include prefix, if necessary.
        """
        if self.opt['t5_generation_config'] is not None and 'text' in observation:
            config = TASK_CONFIGS[self.opt['t5_generation_config']]
            try:
                observation.force_set('text', config['prefix'] + observation['text'])
            except AttributeError:
                observation['text'] = config['prefix'] + observation['text']

        return super().observe(observation)

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output with beam search.

        Use HF's built-in generation to perform beam search.
        """
        bad_words_ids = None
        if self.beam_block_list is not None:
            bad_words_ids = [
                gram for _, ngram in self.beam_block_list.items() for gram in ngram
            ]

        method = self.opt.get('inference', 'greedy')

        generation_params = {
            'input_ids': batch.text_vec,
            'max_length': max_ts,
            'min_length': self.beam_min_length,
            'do_sample': self.opt['inference'] in ['topk', 'topp'],
            'early_stopping': None,
            'num_beams': beam_size,
            'temperature': self.temperature,
            'top_k': self.opt['topk'] if method in ['topk', 'delayedbeam'] else None,
            'top_p': self.opt['topp'] if method == 'nucleus' else None,
            'repetition_penalty': None,
            'bad_words_ids': bad_words_ids if bad_words_ids else None,
            'bos_token_id': self.START_IDX,
            'pad_token_id': self.NULL_IDX,
            'eos_token_id': self.END_IDX,
            'length_penalty': self.opt['beam_length_penalty'],
            'no_repeat_ngram_size': self.beam_block_ngram,
            'num_return_sequences': None,
            'attention_mask': batch.text_vec != self.NULL_IDX,
            'decoder_start_token_id': self.NULL_IDX,
        }

        if self.opt['t5_generation_config']:
            config = TASK_CONFIGS[self.opt['t5_generation_config']]
            config.pop('prefix', None)
            generation_params.update(config)
        if overrides:
            generation_params.update(overrides)

        outputs = self.model.t5.generate(**generation_params)
        outputs = [(outputs[i], 0) for i in range(outputs.size(0))]
        return outputs, []
