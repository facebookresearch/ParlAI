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
from typing import Optional, Dict, Any, Tuple
from transformers import T5ForConditionalGeneration

try:
    from transformers.models.t5.modeling_t5 import T5Stack
except ModuleNotFoundError:
    # Prior versions of transformers package do not have T5Stack
    T5Stack = object

from parlai.agents.hugging_face.hugging_face import HF_VERSION
from parlai.agents.hugging_face.dict import T5DictionaryAgent

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, TorchAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.utils.fsdp import is_fsdp


def check_hf_version(v: Tuple[int, int]) -> bool:
    """
    Check that HF version is greater than 4.3.
    """
    main, sub = v
    return main > 4 or (main == 4 and sub >= 3)


def build_t5(opt: Opt) -> T5ForConditionalGeneration:
    if not check_hf_version(HF_VERSION):
        raise RuntimeError('Must use transformers package >= 4.3 to use t5')
    return T5ForConditionalGeneration.from_pretrained(
        opt['t5_model_arch'], dropout_rate=opt['t5_dropout']
    )


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

    def build_model(self) -> 'ParlaiT5Model':
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
        return T5DictionaryAgent(self.opt)

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for T5.

        T5 dict already adds the end token.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = False  # T5 tokenizer takes care of this
        return TorchAgent.vectorize(self, *args, **kwargs)

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

        model = self.model
        if hasattr(self.model, 'module') and not is_fsdp(self.model):
            model = self.model.module
        outputs = model.t5.generate(**generation_params)
        outputs = [(outputs[i], 0, None) for i in range(outputs.size(0))]
        return outputs, []


##############
# T5 Modules #
##############


class ParlaiT5Encoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, padding_idx: Optional[int] = None):
        super().__init__()
        self.stack = encoder
        self.padding_idx = padding_idx
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
    def __init__(self, opt: Opt, decoder: T5Stack, padding_idx: Optional[int] = None):
        super().__init__()
        self.stack = decoder
        self.padding_idx = padding_idx
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

        mask = input != self.padding_idx
        mask[:, 0] = True  # first token is pad

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
        self.encoder = ParlaiT5Encoder(opt, self.t5.get_encoder(), self.pad_idx)
        self.decoder = ParlaiT5Decoder(opt, self.t5.get_decoder(), self.pad_idx)
        self.paralleled = not opt['t5_model_parallel']

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
        Not *quite* sure how to reconcile this with HF.
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
        tensor = tensor * (self.t5.model_dim**-0.5)
        lm_logits = self.t5.lm_head(tensor)
        return lm_logits


###########################################################################
# Task-specific generation configs for T5.                                #
# Taken from HF: https://huggingface.co/t5-small/resolve/main/config.json #
###########################################################################

TASK_CONFIGS = {
    "summarization": {
        "early_stopping": True,
        "length_penalty": 2.0,
        "max_length": 200,
        "min_length": 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "prefix": "summarize: ",
    },
    "translation_en_to_de": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to German: ",
    },
    "translation_en_to_fr": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to French: ",
    },
    "translation_en_to_ro": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to Romanian: ",
    },
}
