#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Expanded Attention Transformer Model.

Specifically, the decoder uses an additional attention mechanism over a (possibly
separate) context.
"""
import os
import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, List

from parlai.agents.transformer.modules import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerGeneratorModel,
    LAYER_NORM_EPS,
)
from parlai.agents.transformer.modules.decoder import DecoderIncrState
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import create_agent_from_model_file
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, TorchAgent, Output
import parlai.utils.logging as logging
from parlai.utils.torch import PipelineHelper, neginf

from projects.light_whoami.agents.poly_return_weights import (
    PolyencoderReturnCodeWeightsAgent,  # type: ignore
)
from projects.light_whoami.agents.multi_objective import (
    GenerativeMultiObjectiveModule,
    MultiObjectiveGeneratorAgent,
    MultiobjectiveModuleBase,
    TransformerGeneratorReturnLatentModel,
)
from projects.light_whoami.agents.pacer import LongPacerAgent, PacerAgent
from projects.light_whoami.agents.rpa_rerank import LongRPARerankAgent, RPARerankAgent
from projects.light_whoami.task.utils import WHO_AM_I, CONTEXT_KEYS, extract_characters
from projects.msc.agents.long_tga import TransformerVariantAgent, ShiftInvariantEncoder


MaskOut = Union[torch.BoolTensor, torch.Tensor]
Encoding = torch.Tensor
AttnWeights = torch.Tensor

EncoderOutput = Tuple[Encoding, MaskOut, AttnWeights]

ExtraOutput = Union[EncoderOutput, Tuple[Encoding, MaskOut, AttnWeights, AttnWeights]]


def get_classifier_model_and_dict(
    opt: Opt,
) -> Tuple[Optional[TorchAgent], Optional[DictionaryAgent]]:
    """
    Build classifier model and dictionary.
    """
    model_file = modelzoo_path(
        opt['datapath'], opt['expanded_attention_classifier_model_file']
    )
    model, dictionary = None, None
    if model_file and os.path.exists(model_file):
        logging.info(f'Building polyencoder from path: {model_file}')
        logging.disable()
        overrides = {
            'model': 'return_code_weights_agent',
            'data_parallel': opt.get('data_parallel', False),
            'model_parallel': opt['model_parallel'],
            'delimiter': opt['delimiter'],
            'no_cuda': opt['no_cuda'],
            'fp16': opt['fp16'],
        }
        poly_agent = create_agent_from_model_file(model_file, overrides)
        logging.enable()
        logging.info('Poly Build Complete')
        dictionary = poly_agent.build_dictionary()
        model = poly_agent.model
    return model, dictionary


def get_topk(opt: Opt, max_len: int):
    n_input = opt['automated_expanded_attention_n_tokens']
    assert n_input > 0
    topk = min(n_input, max(max_len - 1, 1))
    return topk


class ExpandedDecoderAttentionAgent(TransformerGeneratorAgent):
    """
    The Expanded Decoder agent allows for the decoder layers to attend over multiple
    inputs, separately.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerGeneratorAgent.add_cmdline_args(
            parser, partial_opt=partial_opt
        )  # add transformer args
        expanded_attn = parser.add_argument_group('Expanded Attention Arguments')
        expanded_attn.add_argument(
            '--expanded-attention-init-weights',
            type=str,
            default='random',
            choices=['random', 'encoder_attention'],
            help='how to initialize the expanded attention module.\n'
            'random: initialize weights randomly\n'
            'encoder_attention: initialize weights from encoder_attention weights\n',
        )
        expanded_attn.add_argument(
            '--expanded-attention-share-weights',
            type='bool',
            default=True,
            help='If true, share the weights from the normal encoder attention '
            'with the extra attention (reduce network parameters).',
        )
        expanded_attn.add_argument(
            '--expanded-attention-input-key',
            type=str,
            default='full_text',
            help='Key in observation from which to extract the expanded attn input',
        )
        expanded_attn.add_argument(
            '--expanded-attention-self-character-key',
            type=str,
            default='self_character',
            help='Key in observation with model\'s character',
        )
        expanded_attn.add_argument(
            '--expanded-attention-input-extractor-phrases',
            type=str,
            default='_self_name,_self_persona,_partner_name',
            help='How to extract expanded attention input from the observation\n'
            'Default are phrases from LIGHT',
        )
        expanded_attn.add_argument(
            '--expanded-attention-type',
            type=str,
            choices=['profile', 'automated_classifier', 'automated_trainable_mask'],
            default='profile',
            help='Method of incorporating expanded attention. `profile` manually extracts from '
            'provided context in `--expanded-attention-input-key`, while `automated` uses a separate '
            'classifier to choose the tokens to which we re-attend.',
        )
        expanded_attn.add_argument(
            '--expanded-attention-num-rounds',
            type=int,
            default=1,
            help='how many rounds to re-apply expanded attention',
        )
        ### Classifier Attention
        expanded_attn.add_argument(
            '--automated-expanded-attention-n-tokens',
            type=int,
            default=125,
            help='If using automated attention, how many tokens do we re-attend to.',
        )
        expanded_attn.add_argument(
            '--expanded-attention-classifier-model-file',
            type=str,
            default=None,
            help='specified classifier model if using classifier attention to choose tokens',
        )
        expanded_attn.add_argument(
            '--expanded-attention-classifier-truncate',
            type=int,
            default=510,
            help='Truncation for the classifier.',
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        self.classifier_expanded_attn = (
            opt['expanded_attention_type'] == 'automated_classifier'
        )
        self.trainable_mask_expanded_attn = (
            opt['expanded_attention_type'] == 'automated_trainable_mask'
        )
        self.automated_expanded_attn = (
            self.classifier_expanded_attn or self.trainable_mask_expanded_attn
        )
        super().__init__(opt, shared)
        if shared:
            self.classifier_dict = shared['classifier_dict']
        else:
            _, self.classifier_dict = get_classifier_model_and_dict(opt)

        self.extractor_phrases = []
        if opt['expanded_attention_input_extractor_phrases']:
            self.extractor_phrases = self.opt[
                'expanded_attention_input_extractor_phrases'
            ].split(",")
            assert all(p in CONTEXT_KEYS for p in self.extractor_phrases)

    def share(self):
        shared = super().share()
        shared['classifier_dict'] = self.classifier_dict
        return shared

    def build_model(self, states=None):
        """
        Substitute normal encoder/decoder with custom modules.
        """
        wrapped_class = TransformerExpandedDecoderModel.with_components(
            encoder=TransformerDoubleEncoder,
            decoder=TransformerExpandedDecoder.with_components(
                layer=ExpandedAttentionTransformerDecoderLayer
            ),
        )
        return wrapped_class(self.opt, self.dict)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load the state dict into model.

        Override TA.load_state_dict to build the expanded attention
        """
        try:
            super().load_state_dict(state_dict)
        except RuntimeError:
            if not [k for k in state_dict if 'extra_input_attention' in k]:
                if self.opt['expanded_attention_init_weights'] == 'random':
                    logging.info('Loading Random Init for Expanded Attention')
                    state_dict.update(
                        {
                            **{
                                k: v
                                for k, v in self.model.state_dict().items()
                                if 'extra_input_attention' in k
                            },
                            **{
                                k: v
                                for k, v in self.model.state_dict().items()
                                if 'extra_input_norm' in k
                            },
                        }
                    )
                elif self.opt['expanded_attention_init_weights'] == 'encoder_attention':
                    logging.info('Loading Encoder Attention for Expanded Attention')
                    state_dict.update(
                        {
                            **{
                                k.replace(
                                    'encoder_attention', 'extra_input_attention'
                                ): v
                                for k, v in state_dict.items()
                                if 'decoder' in k and 'encoder_attention' in k
                            },
                            **{
                                k.replace('norm2', 'extra_input_norm'): v
                                for k, v in state_dict.items()
                                if 'decoder' in k and 'norm2' in k
                            },
                        }
                    )
            if not [k for k in state_dict if 'classifier_model' in k]:
                logging.info('Adding Classifier Model Weights')
                state_dict.update(
                    {
                        k: v
                        for k, v in self.model.state_dict().items()
                        if 'classifier_model' in k
                    }
                )
            if not [k for k in state_dict if 'mask_linear' in k]:
                logging.info('Adding trainable mask Weights')
                state_dict.update(
                    {
                        k: v
                        for k, v in self.model.state_dict().items()
                        if 'mask_linear' in k
                    }
                )
            super().load_state_dict(state_dict)

    def observe(self, observation: Union[Dict, Message]) -> Message:
        observation = super().observe(observation)
        observation = self.expanded_observe(observation)
        return observation

    def expanded_observe(self, observation: Message) -> Message:
        """
        Add the relevant expanded input vectors; essentially extrapolates observe, for
        potential subclasses.
        """
        if 'text_vec' not in observation:
            return observation
        if 'expanded_attn_input_vec' not in observation:
            self._set_expanded_attn_input_vec(observation)
            if self.classifier_expanded_attn:
                assert len(observation['text_vec']) == len(
                    observation['expanded_attn_input_vec']
                ), (
                    len(observation['text_vec']),
                    len(observation['expanded_attn_input_vec']),
                )
        if 'character_vec' not in observation:
            self._set_character_vec(observation)
        return observation

    def _set_character_vec(self, observation: Message) -> Message:
        """
        Tokenize the model's character.

        :param observation:
            observation with the character text

        :return observation:
            return observation with the character tokenized.
        """
        if self.opt['expanded_attention_self_character_key'] not in observation:
            return observation
        character = observation[self.opt['expanded_attention_self_character_key']]
        if self.classifier_expanded_attn:
            assert isinstance(self.classifier_dict, DictionaryAgent)
            assert self.classifier_dict is not None
            dictionary = self.classifier_dict
        else:
            dictionary = self.dict
        if '_self_name' in character:
            # We need to extract from the context
            character = extract_characters(character)['_self_name']
        observation['character_vec'] = dictionary.txt2vec(character)
        return observation

    def _set_expanded_attn_input_vec(self, observation: Message) -> Message:
        """
        Tokenize the Expanded Attention Input.

        :param observation:
            observation with input text.

        :return observation:
            return observation with expanded_attn_input_vec.
        """
        expanded_attn_input_vec = None
        if self.opt['expanded_attention_input_key'] not in observation:
            return observation
        expanded_attn_inputs = observation[self.opt['expanded_attention_input_key']]
        if self.extractor_phrases:
            # extract from context
            delim = self.opt.get('delimiter', '\n')
            expanded_attn_inputs = [
                i for ii in expanded_attn_inputs.split('\n') for i in ii.split(delim)
            ]
            expanded_attn_inputs = [
                e
                for e in expanded_attn_inputs
                if any([phrase in e for phrase in self.extractor_phrases])
            ]
            expanded_attn_inputs = delim.join(expanded_attn_inputs)
        if expanded_attn_inputs:
            expanded_attn_input_vec = self._get_expanded_attn_input_vec(
                observation, expanded_attn_inputs
            )
        observation['expanded_attn_input_vec'] = expanded_attn_input_vec
        return observation

    def _get_expanded_attn_input_vec(
        self, observation: Message, expanded_attn_inputs: str
    ) -> List[int]:
        """
        Extract the expanded attention input from the expanded attention inputs string.

        :param expanded_attn_inputs:
            the expanded attention inputs in string form.

        :return expanded_attn_input_vec:
            returns a tokenized version of the expanded attnetion inputs.
        """
        if self.classifier_expanded_attn:
            assert isinstance(self.classifier_dict, DictionaryAgent)
            assert self.classifier_dict is not None
            assert self.classifier_dict.tokenizer == self.dict.tokenizer
            expanded_attn_input_vec = observation['text_vec'].tolist()[
                :-1
            ] + self.classifier_dict.txt2vec(WHO_AM_I)
            return expanded_attn_input_vec[
                -self.opt['expanded_attention_classifier_truncate'] :
            ]
        else:
            dictionary = self.dict
            assert isinstance(dictionary, DictionaryAgent)
            expanded_attn_input_vec = self._check_truncate(
                dictionary.txt2vec(expanded_attn_inputs),
                self.text_truncate,
                False,  # Note: This means we're right-truncating
            )

        return expanded_attn_input_vec

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Overrides TGA.batchify to add expanded input vec.
        """
        batch = super().batchify(obs_batch, sort)
        valid_exs = [ex for ex in obs_batch if self.is_valid(ex)]
        batch.expanded_attn_input_vec = None
        batch.character_vec = None
        if any(ex.get('expanded_attn_input_vec') for ex in valid_exs):
            _extras = [
                ex.get('expanded_attn_input_vec', self.EMPTY) for ex in valid_exs
            ]
            expanded_attn_input_vecs, _lens = self._pad_tensor(_extras)
            batch.expanded_attn_input_vec = expanded_attn_input_vecs
        if any(ex.get('character_vec') for ex in valid_exs):
            _chars = [ex.get('character_vec', self.EMPTY) for ex in valid_exs]
            char_vecs, _lens = self._pad_tensor(_chars)
            batch.character_vec = char_vecs
        return batch

    def _model_input(
        self, batch: Batch
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Pass the expanded attention input vectors through as well.
        """
        return (batch.text_vec, batch.expanded_attn_input_vec, batch.character_vec)


class LongExpandedDecoderAttentionAgent(ExpandedDecoderAttentionAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        ExpandedDecoderAttentionAgent.add_cmdline_args(parser, partial_opt)
        TransformerVariantAgent.add_cmdline_args(parser, partial_opt)
        return parser

    def build_model(self, states=None):
        wrapped_class = TransformerExpandedDecoderModel.with_components(
            encoder=LongTransformerDoubleEncoder,
            decoder=TransformerExpandedDecoder.with_components(
                layer=ExpandedAttentionTransformerDecoderLayer
            ),
        )
        return wrapped_class(self.opt, self.dict)


###########################################
###########################################
#                Modules                  #
###########################################
###########################################
class TransformerExpandedDecoderModel(TransformerGeneratorModel):
    """
    Model with expanded decoder.

    The model uses both a modified encoder (which encodes the expanded attention input)
    as well as a modified decoder (to perform attention over that extra input).
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        super().__init__(opt, dictionary, **kwargs)
        classifier_model = None
        if self.opt['expanded_attention_type'] == 'automated_classifier':
            classifier_model, _ = get_classifier_model_and_dict(self.opt)
            assert isinstance(classifier_model, torch.nn.Module)
            classifier_model.requires_grad_(False)  # by default no learning, for now :)
        self.encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            encoder_class=self.swappables.encoder,  # type: ignore
            classifier_model=classifier_model,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc_out, extra_out = encoder_states
        device = enc_out[0].device
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(device)

        def select_indices(old_out):
            new_out = []
            for st in old_out:
                if isinstance(st, torch.Tensor):
                    new_out.append(torch.index_select(st, 0, indices))
                else:
                    new_out.append(st)
            return new_out

        new_enc_out = select_indices(enc_out)
        new_ext_out = select_indices(extra_out)

        return new_enc_out, new_ext_out


class TransformerDoubleEncoder(TransformerEncoder):
    """
    The Double Encoder encodes both the context and additional expanded attn input
    information.
    """

    def __init__(self, opt: Opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)
        self.classifier_model = kwargs.get('classifier_model')
        if self.opt['expanded_attention_type'] == 'automated_trainable_mask':
            self.mask_linear = torch.nn.Linear(self.dim, self.dim)
            torch.nn.init.xavier_uniform_(self.mask_linear.weight)
            self.mask_dropout = torch.nn.Dropout(self.dropout_frac)
            self.softmax = F.softmax

    def get_extra_output_from_mask(
        self,
        input: torch.LongTensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> ExtraOutput:
        """
        Use a trainable mask layer to determine which elements of the input to re-attend
        to.

        :param input:
            vectorized input tokens
        :param encoder_out:
            output encodings of input tokens
        :param encoder_mask:
            mask for input

        :return (enc_out, enc_mask):
            return the extra output to which we will be attending (for all layers).
        """
        weights = self.softmax(
            self.mask_dropout(self.mask_linear(encoder_output)).masked_fill_(
                (encoder_mask == 0)
                .view(*encoder_mask.size(), 1)
                .expand(*encoder_output.size()),
                neginf(encoder_output.dtype),
            ),
            dim=1,
        )
        topk = get_topk(self.opt, input.size(-1))
        topk_inds = weights.sum(-1).topk(topk, dim=-1, sorted=False).indices
        new_input = torch.gather(input, dim=-1, index=topk_inds)
        out2 = super().forward(new_input)

        assert isinstance(out2, tuple)
        return (*out2, weights)  # type: ignore

    def get_extra_output_from_classifier(
        self, input: torch.LongTensor, attn_weights: torch.Tensor
    ) -> EncoderOutput:
        """
        Use the classifier attention weights (self attn or code attn) to get new
        context.

        :param input:
            original input
        :param attn_weights:
            classifier attn weights

        :return out2:
            return re-encoded extra context to attend to.
        """
        topk = get_topk(self.opt, input.size(-1))
        weights = attn_weights.max(1).values
        assert weights.dim() == 2
        topk_inds = weights.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        new_input = torch.gather(input, dim=-1, index=topk_inds)
        out2 = super().forward(new_input)
        return out2

    def forward(
        self,
        input: torch.LongTensor,
        expanded_attn_inputs: torch.LongTensor = None,
        characters: torch.LongTensor = None,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        ignore_expanded_attn_inputs: bool = False,
    ) -> Tuple[EncoderOutput, ExtraOutput]:
        """
        Encoder forward.

        Encodes both the original input, as well as the input to the decoder expanded attention.

        :param input:
            input to the encoder
        :param expanded_attn_inputs:
            expanded decoder attn input.
        :param characters:
            bsz-length list of characters to encode. used in automated grounding.

        :return (out1, out2):
            out1 is the encoded input
            out2 is the encoded expanded attention input.
        """
        out1 = super().forward(input, positions=positions, segments=segments)
        assert expanded_attn_inputs is not None or ignore_expanded_attn_inputs
        if (
            'automated' in self.opt['expanded_attention_type']
            and expanded_attn_inputs is not None
        ):
            if self.opt['expanded_attention_type'] == 'automated_classifier':
                assert self.classifier_model is not None
                ctxt_rep, ctxt_weights, ctxt_rep_mask, cand_emb = self.classifier_model(
                    ctxt_tokens=expanded_attn_inputs, cand_tokens=characters
                )
                assert cand_emb is not None
                ctxt_char_weights, _ = self.classifier_model(
                    ctxt_rep=ctxt_rep, ctxt_rep_mask=ctxt_rep_mask, cand_rep=cand_emb
                )
                ctxt_weights = torch.bmm(ctxt_char_weights, ctxt_weights)
                out2 = self.get_extra_output_from_classifier(input, ctxt_weights)
            else:
                assert self.opt['expanded_attention_type'] == 'automated_trainable_mask'
                # select extra output from original output; optionally re-encode.
                out2 = self.get_extra_output_from_mask(input, *out1)
        elif expanded_attn_inputs is not None and expanded_attn_inputs.dim() == 3:
            bsz, n_extra, seq_len = expanded_attn_inputs.size()
            out2 = (
                super()
                .forward(expanded_attn_inputs.view(bsz * n_extra, seq_len))
                .view(bsz, n_extra, seq_len, -1)
            )
        elif expanded_attn_inputs is not None:
            out2 = super().forward(expanded_attn_inputs)
        else:
            out2 = out1
        return out1, out2


class LongTransformerDoubleEncoder(ShiftInvariantEncoder, TransformerDoubleEncoder):
    pass


class TransformerExpandedDecoder(TransformerDecoder):
    def forward(
        self,
        input: torch.Tensor,
        encoder_state,
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Override TD.Forward to include extra encoder outputs.
        """
        encoder_output, extra = encoder_state
        extra_output, extra_mask, *_ = extra
        return super().forward(
            input,
            encoder_output,
            incr_state,
            extra_output=extra_output,
            extra_mask=extra_mask,
            **kwargs,
        )

    def forward_layers(
        self,
        tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: DecoderIncrState,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Override to pass more options to model parallel (which is unfortunately not
        handled in super class.)
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel_with_extra(
                tensor, encoder_output, encoder_mask, incr_state, **kwargs
            )
        else:
            tensor, new_incr_state = super().forward_layers(
                tensor, encoder_output, encoder_mask, incr_state=incr_state, **kwargs
            )

        return tensor, new_incr_state

    def _apply_model_parallel_with_extra(
        self,
        tensor,
        encoder_output,
        encoder_mask,
        incr_state,
        extra_output: torch.Tensor = None,
        extra_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Copy paste from TransformerDecoder._apply_model_parallel while incorporating the
        extra output/extra mask.
        """
        chunks = PipelineHelper.split(
            (tensor, encoder_output, encoder_mask, incr_state, extra_output, extra_mask)
        )
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        new_incr_state = {i: [] for i, _ in enumerate(self.layers)}

        for chunk_idx, layer_nos, next_device in work_items:
            (
                s_tensor,
                s_enc_out,
                s_enc_mask,
                s_incr_state,
                s_extra_out,
                s_extra_mask,
            ) = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor, nis = self.layers[layer_no](
                    x=s_tensor,
                    encoder_output=s_enc_out,
                    encoder_mask=s_enc_mask,
                    incr_state=s_incr_state.get(layer_no),
                    extra_output=s_extra_out,
                    extra_mask=s_extra_mask,
                )
                new_incr_state[layer_no].append(nis)
            # don't move incr state, it's always on the correct device
            (
                s_tensor,
                s_enc_out,
                s_enc_mask,
                s_extra_out,
                s_extra_mask,
            ) = PipelineHelper.chunk_to(
                (s_tensor, s_enc_out, s_enc_mask, s_extra_out, s_extra_mask),
                next_device,
            )
            chunks[chunk_idx] = (
                s_tensor,
                s_enc_out,
                s_enc_mask,
                s_incr_state,
                s_extra_out,
                s_extra_mask,
            )

        tensor_out = PipelineHelper.join([c[0] for c in chunks])
        new_incr_state = {
            layer_no: PipelineHelper.join(pieces)
            for layer_no, pieces in new_incr_state.items()
        }

        return tensor_out, new_incr_state  # type: ignore


class ExpandedAttentionTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        opt: Opt,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__(
            opt,
            opt['n_heads'],
            opt['embedding_size'],
            opt['ffn_size'],
            attention_dropout,
            relu_dropout,
            dropout,
            activation,
            variant,
            **kwargs,
        )
        self.opt = opt
        if not opt['expanded_attention_share_weights']:
            self.extra_input_attention = self.swappables.encoder_attention(  # type: ignore
                opt=self.opt,
                n_heads=opt['n_heads'],
                dim=opt['embedding_size'],
                dropout=attention_dropout,
            )
            self.extra_input_norm = torch.nn.LayerNorm(
                opt['embedding_size'], eps=LAYER_NORM_EPS
            )

    def get_extra_attention_module(self):
        """
        Returns the module used for the extra input attention.

        If we're sharing weights between the encoder attention and extra attention, we
        simply return the encoder attention.
        """
        if self.opt['expanded_attention_share_weights']:
            return self.encoder_attention
        else:
            return self.extra_input_attention

    def get_extra_norm_module(self):
        """
        Returns the module used for the extra input layer norm.

        If we're sharing weights between the encoder attention and extra attention, we
        simply return norm2.
        """
        if self.opt['expanded_attention_share_weights']:
            return self.norm2
        else:
            return self.extra_input_norm

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        extra_output: Optional[torch.Tensor] = None,
        extra_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        We insert the expanded attention such that the layer does the following:

        1. Self-attention
        2. Encoder Attention
        3. Expanded Attention
        4. FFN
        """
        assert extra_output is not None
        assert extra_mask is not None
        ######################################
        # Initial Self-Attention is the same #
        ######################################
        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm1(x)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm1(x)

        residual = x

        #######################################
        # First, attend over encoder output   #
        # Then, attend over knowledge output  #
        #######################################
        # encoder_attn_layer_norm norm 2
        if self.variant == 'prelayernorm':
            x = self.norm2(x)

        ######################
        # Normal Enc Attention
        ######################
        x, final_encoder_attn_incr_state = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            incr_state=incr_state.get('encoder_attn'),
            static_kv=True,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm2(x)
        ############################
        # Expanded Input Attention #
        ############################
        final_extra_attn_incr_states = []
        for attention_round in range(self.opt['expanded_attention_num_rounds']):
            residual = x
            if self.variant == 'prelayernorm':
                x = self.get_extra_norm_module()(x)

            x, final_extra_input_attn_incr_state = self.get_extra_attention_module()(
                query=x,
                key=extra_output,
                value=extra_output,
                mask=extra_mask,
                incr_state=incr_state.get(f'extra_attn_{attention_round}'),
                static_kv=True,
                **kwargs,
            )[:2]
            x = self.dropout(x)  # --dropout
            x = x + residual
            if (
                self.variant == 'aiayn'
                or self.variant == 'xlm'
                or self.variant == 'bart'
            ):
                x = self.get_extra_norm_module()(x)
            final_extra_attn_incr_states.append(final_extra_input_attn_incr_state)
        #############################
        # \End Different Attention  #
        #############################
        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm3(x)
        x = self.ffn(x, **kwargs)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm3(x)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_encoder_attn_incr_state,
            **{
                f'extra_attn_{i}': state
                for i, state in enumerate(final_extra_attn_incr_states)
            },
        }
        return x, new_incr_state

    def reorder_incremental_state(
        self, incremental_state: Dict[str, dict], inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'encoder_attn': self.encoder_attention,
            **{
                f'extra_attn_{i}': self.get_extra_attention_module()
                for i in range(self.opt['expanded_attention_num_rounds'])
            },
        }
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }


###################################################
# Expanded Attention + RPA Re-ranker/PACER Models #
###################################################
class ExpandedDecoderAttentionAndRPARerankerAgent(
    ExpandedDecoderAttentionAgent, RPARerankAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        ExpandedDecoderAttentionAgent.add_cmdline_args(parser, partial_opt)
        RPARerankAgent.add_cmdline_args(parser, partial_opt)
        return parser


class ExpandedDecoderAttentionAndPacerAgent(ExpandedDecoderAttentionAgent, PacerAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        ExpandedDecoderAttentionAgent.add_cmdline_args(parser, partial_opt)
        PacerAgent.add_cmdline_args(parser, partial_opt)
        return parser


class LongExpandedDecoderAttentionAndRPARerankerAgent(
    LongExpandedDecoderAttentionAgent, LongRPARerankAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        LongExpandedDecoderAttentionAgent.add_cmdline_args(parser, partial_opt)
        LongRPARerankAgent.add_cmdline_args(parser, partial_opt)
        return parser


class LongExpandedDecoderAttentionAndPacerAgent(
    LongExpandedDecoderAttentionAgent, LongPacerAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        LongExpandedDecoderAttentionAgent.add_cmdline_args(parser, partial_opt)
        LongPacerAgent.add_cmdline_args(parser, partial_opt)
        return parser


########################################
# Expanded Attention + Multi-Objective #
########################################


class ExpandedDecoderAttentionAndMultiObjectiveAgent(
    MultiObjectiveGeneratorAgent, ExpandedDecoderAttentionAgent
):
    """
    Same as the normal all-in-one generator agent, but makes use of the expanded decoder
    models.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        MultiObjectiveGeneratorAgent.add_cmdline_args(parser, partial_opt)
        ExpandedDecoderAttentionAgent.add_cmdline_args(parser, partial_opt)

    def build_model(self) -> MultiobjectiveModuleBase:
        wrapped_class = ExpandedTransformerGeneratorReturnLatentModel.with_components(
            encoder=TransformerDoubleEncoder,
            decoder=TransformerExpandedDecoder.with_components(
                layer=ExpandedAttentionTransformerDecoderLayer
            ),
        )
        model = wrapped_class(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return ExpandedGenerativeMultiObjectiveModule(
            model, self.opt, self.dict, self.NULL_IDX
        )

    def observe(self, observation: Union[Dict, Message]) -> Message:
        """
        Override observe to do agent-specific observation for AIO and expanded decoder.
        """
        observation = super().observe(observation)
        observation = self.expanded_observe(observation)
        observation = self.multiobj_observe(observation)
        return observation


class ExpandedTransformerGeneratorReturnLatentModel(
    TransformerGeneratorReturnLatentModel, TransformerExpandedDecoderModel
):
    pass


class ExpandedGenerativeMultiObjectiveModule(GenerativeMultiObjectiveModule):
    def forward(
        self,
        *xs,
        ys=None,
        prev_enc=None,
        maxlen=None,
        bsz=None,
        multiobjective=False,
        **encoder_kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Override forward to ignore extra input computation for whoami token rep.
        """
        if not multiobjective:
            """
            Copied from TGM.Forward.

            We want to save the latent hidden reps from the decoder.
            """
            return self.base_seq2seq(
                *xs, ys=ys, prev_enc=prev_enc, maxlen=maxlen, bsz=bsz
            )
        else:
            decoder_output, encoder_states = xs
            enc_out, ext_out = encoder_states
            encoder_output, encoder_mask, *_ = enc_out
            bsz, seqlen, _ = decoder_output.size()
            if self.latent_rep == 'decoder_final_layer':
                latent_rep = decoder_output
                latent_mask = None
            elif self.latent_rep == 'encoder_final_layer':
                latent_rep = encoder_output
                latent_mask = encoder_mask
                latent_rep = encoder_output * ext_out[-1]
            elif self.latent_rep == 'encoder_and_decoder':
                whoami_rep, _ = self.base_seq2seq.encoder(
                    self.whoami.unsqueeze(0).expand(bsz, -1),
                    ignore_expanded_attn_inputs=True,
                )
                if isinstance(whoami_rep, tuple):
                    whoami_rep, *_ = whoami_rep
                if not decoder_output.requires_grad:
                    whoami_rep.detach_()
                latent_rep = torch.cat(
                    [encoder_output, whoami_rep, decoder_output], dim=1
                )
                latent_mask = torch.cat(
                    [
                        encoder_mask,
                        encoder_mask.new(bsz, seqlen + whoami_rep.size(1)).fill_(True),
                    ],
                    dim=1,
                )  # type: ignore
            else:
                raise TypeError(
                    f'Latent Representation Not Supported: {self.latent_rep}'
                )
            scores = self.score(latent_rep, latent_mask, ys)  # type: ignore
            return (scores,)


class LongExpandedDecoderAttentionAndMultiObjectiveAgent(
    ExpandedDecoderAttentionAndMultiObjectiveAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        ExpandedDecoderAttentionAndMultiObjectiveAgent.add_cmdline_args(
            parser, partial_opt
        )
        TransformerVariantAgent.add_cmdline_args(parser, partial_opt)
        return parser

    def build_model(self, states=None):
        wrapped_class = ExpandedTransformerGeneratorReturnLatentModel.with_components(
            encoder=LongTransformerDoubleEncoder,
            decoder=TransformerExpandedDecoder.with_components(
                layer=ExpandedAttentionTransformerDecoderLayer
            ),
        )
        model = wrapped_class(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return ExpandedGenerativeMultiObjectiveModule(
            model, self.opt, self.dict, self.NULL_IDX
        )
