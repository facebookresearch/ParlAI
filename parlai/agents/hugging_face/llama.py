#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

import torch
from parlai.agents.hugging_face.dict import LlamaDictionaryAgent, _init_llama_path
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.utils.torch import IdentityLayer, padded_tensor
from parlai.agents.transformer.transformer import _check_positional_embeddings
import parlai.utils.logging as logging

try:
    from transformers import LlamaModel, LlamaForCausalLM
except ImportError:
    raise ImportError("Please run `pip install transformers --upgrade`.")


def setup_llama_args(parser):
    if parser is None:
        return parser
    parser.add_argument(
        "--llama-model-dir",
        type=str,
        default=None,
        help="dir to llama model and tokenizer",
    )
    parser.add_argument(
        "--llama-tokenizer-fast",
        type=bool,
        default=True,
        help="whether to load llama fast tokenizer, LlamaTokenizerFast is slower to load but fast in tokenization",
    )
    parser.set_defaults(
        text_truncate=1792,
        label_truncate=256,
        dict_maxexs=0,  # skip building dictionary
    )
    return parser


############################################
## Modules
############################################


class ParlaiLlamaDecoder(torch.nn.Module):
    """
    Llama Decoder.

    This decoder is initialized with the pretrained model from Hugging Face.
    """

    def __init__(
        self,
        opt: Opt,
        decoder: LlamaModel,
        padding_idx: Optional[int] = None,
        start_idx: Optional[int] = None,
    ):
        super().__init__()
        self.model = decoder
        self.padding_idx = padding_idx
        self.start_idx = start_idx

    def forward(
        self, input, encoder_state, incr_state=None, sequence_classification=False
    ):
        attention_mask = None
        position_ids = None
        input_cut_start_token = False
        if sequence_classification:
            assert incr_state is None
        if incr_state is None:
            if sequence_classification:
                # for sequence classification, ignore the input (is None anyway)
                model_input = encoder_state
            elif input.size(1) == 1 and (input[:, 0] == self.start_idx).all():
                # first step generating: ignore the start token
                model_input = encoder_state
                input_cut_start_token = True
            else:
                # forced decoding: concatenate the context with the labels
                # cut decoder_start_token:
                if (input[:, 0] == self.start_idx).all():
                    input = input[:, 1:]
                    input_cut_start_token = True
                model_input = torch.cat([encoder_state, input], dim=-1)

            attention_mask = model_input != self.padding_idx
            position_ids = (
                attention_mask.cumsum(dim=-1, dtype=torch.int64) - 1
            ).clamp_(min=0)
        else:
            # generating with continuation
            # cut decoder_start_token
            if (input[:, 0] == self.start_idx).all():
                input = input[:, 1:]
                input_cut_start_token = True
            # generation: get the last token input
            model_input = input[:, -1:]
            # get the position ids
            position_ids = (encoder_state != self.padding_idx).sum(
                -1, True, dtype=torch.int64
            ) - 1
            delta = ((input != self.padding_idx)).sum(-1, True, dtype=torch.int64)
            position_ids += delta
            attention_mask = (
                torch.cat([encoder_state, input], dim=-1) != self.padding_idx
            )

        model_input = model_input.clamp_(min=0)
        transformer_outputs = self.model(
            input_ids=model_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=incr_state,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        if sequence_classification is True:
            # return full hidden states for sequence level classification
            output = hidden_states
        elif incr_state is None:
            # pull out only the hidden states for the label tokens
            output = hidden_states[:, -input.size(1) - int(input_cut_start_token) :]
        else:
            # generation, we're only doing one token at a time. no need to shove things back in
            output = hidden_states

        return output, new_incr_state


class ParlaiLlamaForCausalLM(TorchGeneratorModel):
    """
    Hugging Face llama Model.

    Llama is a multi-layer decoder-only Transformer. As such, the encoder
    is simply an identity layer. The decoder is initialized with pretrained
    weights from the original Llama Model. Read more about this model here
    <https://huggingface.co/docs/transformers/main/model_doc/llama>.
    """

    def __init__(self, opt, dict):
        super().__init__(dict.null_idx, dict.start_idx, dict.end_idx)
        # init the model
        llama_path = _init_llama_path(opt)
        logging.info(f'Loading Llama model from {llama_path}')
        llama = LlamaForCausalLM.from_pretrained(llama_path)
        self.encoder = IdentityLayer()
        self.pad_idx = dict.null_idx
        self.decoder = ParlaiLlamaDecoder(
            opt, llama.get_decoder(), self.pad_idx, dict.start_idx
        )
        self.config = llama.config
        self.lm_head = torch.nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self._tie_weights(self.lm_head, llama.get_output_embeddings())

    def _tie_weights(self, lm_head, output_embeddings):
        lm_head.weight = output_embeddings.weight

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
            if torch.is_tensor(layer_past):
                new_incr_state.append(torch.index_select(layer_past, 1, inds))
            else:
                # newer versions of HF split up the intermediate outputs
                assert isinstance(layer_past, tuple)
                layer_past = torch.stack(layer_past, dim=0)
                new_incr_state.append(torch.index_select(layer_past, 1, inds))

        return tuple(new_incr_state)


############################################
## Generative Agent
############################################


class LlamaAgent(TorchGeneratorAgent):
    """
    Hugging Face Llama Agent.

    LlamaForCausalLM is a multi-layer decoder-only Transformer.
    The decoder is initialized with pretrained weights from Hugging Face.
    Read more about this model here
    <https://huggingface.co/docs/transformers/main/model_doc/llama>.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group("LlamaAgent Args")
        group = setup_llama_args(group)
        return parser

    def __init__(self, opt, shared=None):
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
        return LlamaDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return ParlaiLlamaForCausalLM(self.opt, self.dict)

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    def _pad_tensor(self, items, is_label=False):
        """
        Override to always set fp16friendly to False.

        Pads context tensor on the left and label tensor on the right, such that when
        they are concatenated the example meets in the middle to form a continuous
        sequence.
        """
        return padded_tensor(
            items, pad_idx=self.NULL_IDX, left_padded=(not is_label), fp16friendly=False
        )

    def _set_label_vec(self, obs, add_start, add_end, truncate):
        super()._set_label_vec(obs, add_start, add_end, truncate)
        # cut off the start_token in the label_vec if any (llama_tokenizer add_bos_token = True), don't compute the loss!!
        for label_vec_type in ['labels_vec', "eval_labels_vec"]:
            if label_vec_type in obs and obs[label_vec_type][0] == self.START_IDX:
                obs.force_set(label_vec_type, obs[label_vec_type][1:])
        return obs


############################################
## Classifier Agent
############################################


class ParlaiLlamaForSequenceClassification(torch.nn.Module):
    """
    Wrap a llama decoder in a linear layer.
    """

    def __init__(self, opt, dict, num_classes):
        super().__init__()
        llama_path = _init_llama_path(opt)
        logging.info(f'Loading Llama model from {llama_path}')
        llama = LlamaForCausalLM.from_pretrained(llama_path)
        self.base_model = ParlaiLlamaDecoder(
            opt, llama.get_decoder(), dict.null_idx, dict.start_idx
        )
        self.config = llama.config
        self.additional_linear_layer = torch.nn.Linear(
            self.config.hidden_size, num_classes, bias=False
        )

    def forward(self, xs) -> torch.Tensor:
        """
        Forward pass.

        Apply transformer, then additional linear layer.
        return:
            bsz x num_classes
        """
        latent, _ = self.base_model(
            input=None, encoder_state=xs, sequence_classification=True
        )
        # latent: [bsz * seqlen * emb_dim]
        return self.additional_linear_layer(latent[:, -1:].squeeze(dim=1))


class ParlaiLlamaClassifierAgent(TorchClassifierAgent):
    """
    Hugging Face LlamaForSequenceClassification.

    Llama is a multi-layer decoder-only Transformer. As such, the encoder
    is simply an identity layer. The decoder is initialized with pretrained
    weights from Hugging Face. Read more about this model here
    <https://huggingface.co/docs/transformers/main/model_doc/llama>.

    parlai em -m parlai.agents.hugging_face.llama:ParlaiLlamaClassifierAgent --classes __notok__ __ok__
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add CLI args.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('ParlaiLlamaClassifierAgent Arguments')
        # class arguments
        group = setup_llama_args(group)
        return parser

    def build_model(self):
        _check_positional_embeddings(self.opt)
        num_classes = len(self.class_list)
        return ParlaiLlamaForSequenceClassification(self.opt, self.dict, num_classes)

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overridden if a more complex dictionary is required.
        """
        return LlamaDictionaryAgent

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)

        if 'text_vec' in obs and 'added_start_end' not in obs:
            obs.force_set(
                'text_vec',
                self._add_start_end_tokens(
                    obs['text_vec'],
                    add_start=(not self.dict.add_special_tokens),
                    add_end=False,
                ),
            )
            obs['added_start_end'] = True

        # check truncation after adding start end tokens
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(
                obs['text_vec'], self.text_truncate, truncate_left=True
            )
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))

        return obs

    def score(self, batch):
        return self.model(batch.text_vec)
