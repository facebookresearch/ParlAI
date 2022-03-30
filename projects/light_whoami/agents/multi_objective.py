#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Multi-objective Agents.

These agents generally contain extra layers to solve the Am I Me or You task.
"""
from abc import ABC
import copy
import random
import torch
import torch.nn
from typing import Optional, Tuple, Union, Type, List, Dict, Any

from parlai.agents.transformer.modules import (
    TransformerEncoder,
    TransformerGeneratorModel,
)
from parlai.agents.transformer.polyencoder import PolyBasicAttention, PolyencoderAgent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import TorchAgent, Batch
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorModel
import parlai.utils.logging as logging
from parlai.utils.misc import recursive_getattr, warn_once

from projects.light_whoami.task.utils import WHO_AM_I
from projects.light_whoami.agents.rpa_rerank import RPAReranker
from projects.msc.agents.long_tga import TransformerVariantAgent, ShiftInvariantEncoder


VALID_MULTIOBJ_LOSSES = ['full', 'sliced', 'partial']


class MultiObjectiveAgentBase(TorchAgent, ABC):
    """
    Define API for All-in-one Agents.

    _base_model_key is the key in the state dict mapping to the normal, "base" model.
    _cand_emb_key is the key in the state dict from which one would initialize the
    candidate encoder embeddings (if copying the embeddings).
    """

    _base_model_key: str
    _cand_emb_key: str

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('MultiObjective Arguments')
        agent.add_argument(
            '--n-multiobjective-heads',
            type=int,
            default=16,
            help='how many attention heads for the multiobjective transformer',
        )
        agent.add_argument(
            '--n-multiobjective-layers',
            type=int,
            default=2,
            help='how many attention layers for the multiobjective transformer',
        )
        agent.add_argument(
            '--copy-cand-embeddings',
            type='bool',
            default=True,
            help='If true, copy embeddings from the base model candidate encoder '
            'to the extra cand layers embeddings',
        )
        return parser

    def build_model(self):
        raise RuntimeError('Need to re-implement build_model for MultiObjective Agents')

    @property
    def base_model_key(self) -> str:
        assert self._base_model_key
        return self._base_model_key

    @property
    def cand_emb_key(self) -> str:
        assert self._cand_emb_key
        return self._cand_emb_key

    def load_state_dict(self, state_dict):
        """
        Overrided to rename initial bases weights appropriately.
        """
        key = self.base_model_key
        if not any(key in k for k in state_dict):
            state_dict = {f"{key}.{k}": v for k, v in state_dict.items()}
            for k, v in self.model.state_dict().items():
                if k not in state_dict:
                    state_dict[k] = v
            if self.opt['copy_cand_embeddings']:
                state_dict = self.update_cand_embeddings(state_dict)

        super().load_state_dict(state_dict)

    def update_cand_embeddings(self, state_dict):
        """
        Update the candidate encoder embeddings, if copying from the base model.
        """
        cand_key = self.cand_emb_key
        all_keys = list(state_dict.keys())
        for k in all_keys:
            if cand_key in k and '.embeddings' in k:
                update_key = k.replace(cand_key, 'extra_layers_cand')
                assert update_key in state_dict, update_key
                state_dict[update_key] = state_dict[k]
        return state_dict

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.

        Switch to `base_model`.
        """
        # map extra special tokens carefully
        key = self.base_model_key
        base_model = getattr(self.model, key)
        new_size = base_model.embeddings.weight.size()[0]
        orig_size = state_dict[f'{key}.embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            'embeddings.weight',
            'encoder.embeddings.weight',
            'decoder.embeddings.weight',
        ]:
            # get new_embs
            old_embs = state_dict[f"{key}.{emb_weights}"]
            new_embs = recursive_getattr(base_model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[f"{key}.{emb_weights}"] = new_embs

        if self.opt['copy_cand_embeddings']:
            state_dict = self.update_cand_embeddings(state_dict)

        return state_dict


class BypassEmbeddingTransformer(TransformerEncoder):
    """
    Transformer Encoder that bypasses embedding phase.
    """

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Skip embedding.
        """
        tensor = input
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        mask = tensor.new_ones(tensor.size()[:2])

        return tensor, mask


class MultiobjectiveModuleBase(torch.nn.Module, ABC):
    """
    Base module for multi-objective layers.

    This module contains context encoding layers, candidate encoding layers, and an
    attention mechanism.
    """

    extra_layers_ctxt: BypassEmbeddingTransformer
    extra_layers_cand: Union[TransformerEncoder, BypassEmbeddingTransformer]
    attention: PolyBasicAttention
    opt: Opt

    def freeze_extra_layers(self):
        """
        Freeze the extra layers of the transformer.
        """
        for parameter in self.extra_layers_cand.parameters():
            parameter.requires_grad = False
        for parameter in self.extra_layers_ctxt.parameters():
            parameter.requires_grad = False

    def unfreeze_extra_layers(self):
        """
        Unfreeze the extra layers of the transformer.
        """
        for parameter in self.extra_layers_cand.parameters():
            parameter.requires_grad = True
        for parameter in self.extra_layers_ctxt.parameters():
            parameter.requires_grad = True

    def get_encoder(
        self,
        opt: Opt,
        dict_: DictionaryAgent,
        embeddings: torch.nn.Embedding,
        module_klass: Type[TransformerEncoder],
        null_idx: int,
        reduction_type: Optional[str] = None,
    ):
        """
        Return encoder, given options.

        Ensures that multiobjective options are copied correctly.

        :param opt:
            opt dict
        :param dict:
            dictionary agent
        :param null_idx:
            null/pad index into dict
        :param reduction_type:
            reduction type for the encoder
        :return:
            a TransformerEncoder, initialized correctly
        """
        opt = copy.deepcopy(opt)
        opt['n_heads'] = opt.get('n_multiobjective_heads', 4)
        opt['n_layers'] = opt.get('n_multiobjective_layers', 2)
        opt['n_encoder_layers'] = opt.get('n_multiobjective_layers', 2)
        opt['n_decoder_layers'] = opt.get('n_multiobjective_layers', 2)
        return module_klass(
            opt=opt,
            vocabulary_size=len(dict_),
            embedding=embeddings,
            padding_idx=null_idx,
            reduction_type=reduction_type,
            n_segments=opt.get('n_segments', 2),
        )

    def score(
        self,
        ctxt_rep: torch.Tensor = None,
        ctxt_rep_mask: torch.BoolTensor = None,
        cand_rep: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute scores given pre-computed context and candidate representations.

        Runs representations through transformers, attends etc.

        :param ctxt_rep:
            context representation as computed by base model
        :param ctxt_rep_mask:
            mask for ctxt rep
        :param cand_rep:
            candidate represetnation as computed by base model

        :return scores:
            return scores after further computation with additional layers
        """
        assert ctxt_rep is not None and cand_rep is not None
        ctxt_rep = self.get_ctxt_rep(ctxt_rep)
        cand_rep = self.get_cand_rep(cand_rep)

        ctxt_final_rep = self.attention(
            cand_rep, ctxt_rep, values=ctxt_rep, mask_ys=ctxt_rep_mask
        )
        scores = torch.sum(ctxt_final_rep * cand_rep, 2)
        return scores

    def get_ctxt_rep(self, ctxt_rep: torch.Tensor) -> torch.Tensor:
        """
        Further encode context representation.
        """
        ctxt_rep, *_ = self.extra_layers_ctxt(ctxt_rep)
        return ctxt_rep

    def get_cand_rep(self, cand_rep: torch.Tensor) -> torch.Tensor:
        """
        Further encode candidate representation.
        """
        bsz, num_cands, _ = cand_rep.size()
        cand_rep, *_ = self.extra_layers_cand(cand_rep.view(bsz * num_cands, -1))
        cand_rep = cand_rep.view(bsz, num_cands, -1)
        return cand_rep


#######################
# Generative-Specific #
#######################


class TransformerGeneratorReturnLatentModel(TransformerGeneratorModel):
    """
    Returns the latent representations in the forward pass.

    We need the latent representations for the multi-objective computations
    """

    def decode_forced(
        self, encoder_states: Tuple[Any], ys: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.BoolTensor]:
        """
        Override TGM.decode_forced to return latent states.

        Nearly copied verbatim, except for return type.
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        latent, mask = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds, latent, mask

    def forward(
        self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.BoolTensor, Any]:
        """
        Get output predictions from the model.

        Nearly copied verbatim, except for return type.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        # use teacher forcing
        scores, preds, latent, mask = self.decode_forced(encoder_states, ys)
        return scores, preds, latent, mask, encoder_states


class GenerativeMultiObjectiveModule(MultiobjectiveModuleBase):
    """
    Wraps a sequence to sequence transformer to compute multi-objective predictions.
    """

    @property
    def encoder(self) -> torch.nn.Module:
        return self.base_seq2seq.encoder

    @property
    def decoder(self) -> torch.nn.Module:
        return self.base_seq2seq.decoder

    def __init__(
        self,
        base_seq2seq: TransformerGeneratorReturnLatentModel,
        opt: Opt,
        dict_: DictionaryAgent,
        null_idx: int,
    ):
        """
        Init additional poly-encoder layers.
        """
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.base_seq2seq = base_seq2seq
        embeddings = self.base_seq2seq.embeddings
        self.latent_rep = opt['multiobjective_latent_representation']
        self.extra_layers_ctxt = self.get_encoder(
            opt, dict_, embeddings, BypassEmbeddingTransformer, null_idx
        )
        self.extra_layers_cand = self.get_encoder(
            opt, dict_, embeddings, TransformerEncoder, null_idx, reduction_type='mean'
        )
        self.attention = PolyBasicAttention(
            opt['polyencoder_type'],
            opt['poly_n_codes'],
            dim=2,
            attn=opt['poly_attention_type'],
            get_weights=False,
        )
        self.whoami = torch.LongTensor(dict_.txt2vec(WHO_AM_I))
        if not opt['no_cuda'] and torch.cuda.is_available():
            self.whoami = self.whoami.to('cuda')
        self.multiobjective_losses = opt['multiobjective_loss'].split(',')
        if opt['multiobjective_loss_ratio'] == 1.0 and all(
            opt[f'multiobjective_{l}_loss_backprop'] == 'extra_layers'
            for l in self.multiobjective_losses
        ):
            # Freeze the base seq2seq model; we're not training it.
            for param in self.base_seq2seq.parameters():
                param.requires_grad = False

    def get_cand_rep(self, cands: torch.LongTensor) -> torch.Tensor:
        """
        Override to account for cand_rep being tokens.

        :param cand_rep:
            here, it's actually the tokenized candidate vectors.
        """
        if cands.dim() == 2:
            bsz, num_cands = cands.size(0), 1
        else:
            assert cands.dim() == 3
            bsz, num_cands, *_ = cands.size()
        cand_rep = self.extra_layers_cand(cands.view(bsz * num_cands, -1))
        return cand_rep.view(bsz, num_cands, -1)

    def forward(
        self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None, multiobjective=False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Overload TGM.forward to account for both seq2seq modeling and multiobjective
        modeling.
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
            encoder_output, encoder_mask = encoder_states
            bsz, seqlen, _ = decoder_output.size()
            if self.latent_rep == 'decoder_final_layer':
                scores = self.score(decoder_output, None, ys)
            elif self.latent_rep == 'encoder_final_layer':
                scores = self.score(encoder_output, encoder_mask, ys)
            elif self.latent_rep == 'encoder_and_decoder':
                multiobjective_rep, _ = self.base_seq2seq.encoder(
                    self.whoami.unsqueeze(0).expand(bsz, -1)
                )
                if isinstance(multiobjective_rep, tuple):
                    multiobjective_rep, _ = multiobjective_rep
                if not decoder_output.requires_grad:
                    multiobjective_rep.detach_()
                latent_rep = torch.cat(
                    [encoder_output, multiobjective_rep, decoder_output], dim=1
                )
                mask = torch.cat(
                    [
                        encoder_mask,
                        encoder_mask.new(
                            bsz, seqlen + multiobjective_rep.size(1)
                        ).fill_(True),
                    ],
                    dim=1,
                )  # type: ignore
                scores = self.score(latent_rep, mask, ys)  # type: ignore
            else:
                raise TypeError(
                    f'Latent Representation Not Supported: {self.latent_rep}'
                )
            return (scores,)

    #################################
    # TorchGeneratorModel Overrides #
    #################################

    def _get_initial_forced_decoder_input(self, *args, **kwargs):
        return self.base_seq2seq._get_initial_forced_decoder_input(*args, **kwargs)

    def reorder_encoder_states(self, *args, **kwargs):
        return self.base_seq2seq.reorder_encoder_states(*args, **kwargs)

    def reorder_decoder_incremental_state(self, *args, **kwargs):
        return self.base_seq2seq.reorder_decoder_incremental_state(*args, **kwargs)

    def output(self, *args, **kwargs):
        return self.base_seq2seq.output(*args, **kwargs)


class MultiObjectiveGeneratorAgent(
    MultiObjectiveAgentBase, TransformerGeneratorAgent, TorchRankerAgent
):
    """
    Add an additional target to the model output representations for predicting model's
    character.
    """

    _base_model_key = 'base_seq2seq'
    _cand_emb_key = 'base_seq2seq.encoder'

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        PolyencoderAgent.add_cmdline_args(parser, partial_opt)
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt)
        MultiObjectiveAgentBase.add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Multi-Objective Generator Arguments')
        agent.add_argument(
            '--character-key',
            type=str,
            default='full_text',
            help='key from which to extract characters',
        )
        agent.add_argument(
            '--character-candidates-key',
            type=str,
            default='full_text',
            help='key from which to extract character candidates (i.e., candidates for ranking)',
        )
        agent.add_argument(
            '--multiobjective-latent-representation',
            type=str,
            default='decoder_final_layer',
            choices=[
                'encoder_final_layer',
                'decoder_final_layer',
                'encoder_and_decoder',
            ],
            help='Which latent representation from the base model to pass to the extra layers',
        )
        agent.add_argument(
            '--multiobjective-loss-ratio',
            type=float,
            default=0.5,
            help='ratio of multiobjective loss to normal loss',
        )
        agent.add_argument(
            '--multiobjective-loss',
            type=str,
            default='full',
            help='Type of Multi-Objective loss to apply to this model, while training: '
            'full ==> use the full teacher-forced output distributions as input to the classifier\n'
            'partial ==> each partial output is classified as its own example. weights of the classifier '
            'are kept constant\n'
            'sliced ==> we arbitrarily slice the full output to compute partial-output losses\n'
            'Can specify multiple via comma-separation.',
        )
        agent.add_argument(
            '--multiobjective-full-loss-backprop',
            type=str,
            default='all',
            choices=['all', 'extra_layers', 'base_model'],
            help='Which parameters to train with the multiobjective full output loss',
        )
        agent.add_argument(
            '--multiobjective-sliced-loss-backprop',
            type=str,
            default='all',
            choices=['all', 'extra_layers', 'base_model'],
            help='Which parameters to train with the multiobjective sliced output loss',
        )
        agent.add_argument(
            '--multiobjective-partial-loss-backprop',
            type=str,
            default='base_model',
            choices=['all', 'extra_layers', 'base_model'],
            help='Which parameters to train with the multiobjective partial output loss',
        )
        agent.add_argument(
            '--partial-output-loss-threshold',
            type=float,
            default=-1,
            help='threshold of full loss at which point partial output loss can be computed'
            'e.g., if set to 10, we dont compute partial loss until full loss reaches 10 or lower',
        )
        agent.add_argument(
            '--sliced-output-loss-threshold',
            type=float,
            default=-1,
            help='threshold of full loss at which point sliced output loss can be computed'
            'e.g., if set to 10, we dont compute sliced loss until full loss reaches 10 or lower',
        )

        return parser

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.multiobjective_losses = opt['multiobjective_loss'].split(',')
        assert all(l in VALID_MULTIOBJ_LOSSES for l in self.multiobjective_losses)
        self.partial_loss_threshold = opt['partial_output_loss_threshold']
        self.sliced_loss_threshold = opt['sliced_output_loss_threshold']
        self.crossed_partial_loss_threshold = self.partial_loss_threshold < 0
        self.crossed_sliced_loss_threshold = self.sliced_loss_threshold < 0

        if shared:
            self.multiobj_criterion = shared['multiobj_criterion']
        else:
            self.multiobj_criterion = TorchRankerAgent.build_criterion(self)

    def share(self):
        shared = super().share()
        shared['multiobj_criterion'] = self.multiobj_criterion
        return shared

    def build_model(self) -> TorchGeneratorModel:
        base_model = TransformerGeneratorReturnLatentModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                base_model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return GenerativeMultiObjectiveModule(
            base_model, self.opt, self.dict, self.NULL_IDX
        )

    def observe(self, observation: Union[Dict, Message]) -> Message:
        """
        Override TA.observe to tokenize characters.
        """
        observation = super().observe(observation)
        observation = self.multiobj_observe(observation)
        return observation

    def multiobj_observe(self, observation: Message) -> Message:
        """
        Tokenize characters and candidates.
        """
        if 'character_vec' not in observation:
            self._set_character_vec(observation)
        if 'character_candidates_vec' not in observation:
            self._set_character_candidates_vec(observation)
        return observation

    def _set_character_vec(self, observation: Message) -> Message:
        """
        Tokenize the character vectors.
        """
        if self.opt['character_key'] not in observation:
            return observation
        character = RPAReranker.get_class_to_rerank_for(
            observation, observation[self.opt['character_key']]
        )
        if character:
            observation['character_vec'] = self.dict.txt2vec(character)
        return observation

    def _set_character_candidates_vec(self, observation: Message) -> Message:
        """
        Tokenize the character candidates.
        """
        if self.opt['character_candidates_key'] not in observation:
            return observation
        characters = observation[self.opt['character_candidates_key']]
        if not isinstance(characters, list):
            characters = RPAReranker.get_predictor_label_candidates(
                observation, characters
            )
        if any(c for c in characters):
            observation['character_candidates_vec'] = [
                torch.LongTensor(self.dict.txt2vec(c)) for c in characters
            ]
        return observation

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        batch = super().batchify(obs_batch, sort)
        valid_exs = [ex for ex in obs_batch if self.is_valid(ex)]
        batch.character_vec = None
        batch.character_candidates = None
        if any(ex.get('character_vec') is not None for ex in valid_exs):
            _chars = [ex.get('character_vec', self.EMPTY) for ex in valid_exs]
            c_vecs, _c_lens = self._pad_tensor(_chars)
            batch.character_vec = c_vecs
        if any(ex.get('character_candidates_vec') is not None for ex in valid_exs):
            batch.character_candidates = [
                ex.get('character_candidates_vec', None) for ex in valid_exs
            ]
        return batch

    def _build_character_candidates(
        self, batch: Batch
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Build set of character candidates from the incoming batch.

        :param batch:
            training/eval batch

        :return cand_vecs, label_inds:
            return [bsz, n_cands] set of candidates, as well as indices within dim 1
            for correct label
        """
        _cands, cand_vecs, label_inds = self._build_candidates(
            Batch(
                batchsize=batch.text_vec.size(0),
                is_training=batch.is_training,
                text_vec=batch.text_vec,
                label_vec=batch.character_vec,
                valid_indices=batch.valid_indices,
                candidate_vecs=batch.character_candidates,
                image=batch.image,
                rewards=batch.rewards,
                observations=batch.observations,
            ),
            self.opt['candidates']
            if batch.is_training
            else self.opt['eval_candidates'],
            mode='compute_loss',
        )
        return cand_vecs, label_inds

    def get_multiobjective_output(
        self,
        latent: torch.Tensor,
        encoder_states: Tuple[torch.Tensor, ...],
        ys: torch.LongTensor,
        loss_type: str,
    ) -> torch.Tensor:
        """
        Compute Multi-Objective Output Scores.

        :param latent:
            decoder output from the base model
        :param encoder_states:
            encoder output from the base model
        :param ys:
            candidate vectors for the character classification task
        :param loss_type:
            which type of multiobjective loss we are computing.

        :return char_scores:
            return scores for each of the characters in ys
        """
        if self.opt[f'multiobjective_{loss_type}_loss_backprop'] == 'base_model':
            self.model.freeze_extra_layers()
        elif self.opt[f'multiobjective_{loss_type}_loss_backprop'] == 'extra_layers':
            latent = latent.detach()
            if isinstance(encoder_states[0], tuple):
                encoder_states[0][0].detach_()
            else:
                assert isinstance(encoder_states[0], torch.Tensor)
                encoder_states[0].detach_()
        char_scores, *_ = self.model(latent, encoder_states, ys=ys, multiobjective=True)
        if self.opt[f'multiobjective_{loss_type}_loss_backprop'] == 'base_model':
            self.model.unfreeze_extra_layers()
        return char_scores

    def compute_partial_decoded_loss(
        self,
        batch: Batch,
        latent: torch.Tensor,
        encoder_states: Tuple[torch.Tensor, ...],
        cand_vecs: torch.LongTensor,
        label_inds: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute partial loss from decoding outputs.

        Here, we consider each partially decoded sequence as a separate
        item from which to compute multiobjective scores.

        :param batch:
            batch being considered
        :param latent:
            decoder output representations
        :param encoder_states:
            encoder output representations
        :param cand_vecs:
            character candidate vectors
        :param label_inds:
            list of indices indicating which character is correct in the character candidates

        :return partial_loss:
            return loss for each batch item as a sum of the partial losses.
        """
        assert self.opt['multiobjective_latent_representation'] == 'decoder_final_layer'
        assert latent.dim() == 3 and latent.size(0) == cand_vecs.size(0)
        bsz, seq_len, dim = latent.size()
        seq_lens = []
        partial_char_losses = []
        seq_scores = []
        stride_length = 2
        for stride in range(0, bsz, stride_length):  # arbitrary stride for now
            # Compute new batches of items; latent reps, candidate vectors, etc.
            end_idx = min(stride + stride_length, bsz)
            new_bsz = batch.label_vec[stride:end_idx].ne(self.NULL_IDX).sum().item()
            new_latent = latent.new(new_bsz, seq_len, dim).fill_(0)
            new_cand_vecs = cand_vecs.new(new_bsz, *cand_vecs.shape[1:]).fill_(
                self.NULL_IDX
            )
            if new_cand_vecs.dim() == 2:
                new_cand_vecs = new_cand_vecs.unsqueeze(1).repeat(
                    1, cand_vecs.size(0), 1
                )
            new_label_inds = label_inds[stride:end_idx].new(new_bsz).fill_(0)

            # For each batch item in the stride, we compute seq_length examples
            # where each example represents a partial output of the decoder.
            offset = 0
            for i in range(stride, end_idx):
                cand_vecs_i = cand_vecs if cand_vecs.dim() == 2 else cand_vecs[i]
                seq_len_i = batch.label_vec[i].ne(self.NULL_IDX).sum().item()
                seq_lens.append(seq_len_i)
                for j in range(seq_len_i):
                    new_latent[offset + j, 0 : j + 1, :] = latent[
                        i : i + 1, 0 : j + 1, :
                    ]
                new_cand_vecs[offset : offset + seq_len_i] = cand_vecs_i
                new_label_inds[offset : offset + seq_len_i] = label_inds[
                    i : i + 1
                ].repeat(seq_len_i)
                offset += seq_len_i

            assert isinstance(new_cand_vecs, torch.LongTensor)
            seq_score = self.get_multiobjective_output(
                new_latent, encoder_states, new_cand_vecs, 'partial'
            )
            partial_char_losses.append(
                self.multiobj_criterion(seq_score, new_label_inds)
            )
            seq_scores.append(seq_score)
        partial_char_loss = torch.cat(partial_char_losses, dim=0)
        seq_scores = torch.cat(seq_scores, dim=0)
        partial_char_loss_metric = partial_char_loss.new(bsz).fill_(0)
        offset = 0
        partial_char_scores = torch.zeros(
            batch.batchsize,
            batch.batchsize if cand_vecs.dim() == 2 else cand_vecs.size(1),
        ).to(latent)
        for i in range(bsz):
            partial_char_loss_metric[i] = partial_char_loss[
                offset : offset + seq_lens[i]
            ].mean()
            partial_char_scores[i] = seq_scores[
                partial_char_loss[offset : offset + seq_lens[i]].argmin()
            ]
        self.compute_multiobj_metrics(
            partial_char_loss_metric, partial_char_scores, label_inds, prefix='partial'
        )
        return partial_char_loss

    def compute_loss(self, batch, return_output=False):
        """
        Override compute_loss for multi-objective loss computation.
        """
        loss, model_output = super().compute_loss(batch, return_output=True)
        _scores, _preds, latent, mask, encoder_states = model_output
        cand_vecs, label_inds = self._build_character_candidates(batch)

        full_char_loss = torch.zeros(batch.batchsize).to(loss)
        sliced_char_loss = torch.zeros(batch.batchsize).to(loss)
        partial_char_loss = torch.zeros(batch.batchsize, batch.label_vec.size(1)).to(
            loss
        )
        if 'full' in self.multiobjective_losses:
            char_scores = self.get_multiobjective_output(
                latent, encoder_states, cand_vecs, 'full'
            )
            full_char_loss = self.multiobj_criterion(char_scores, label_inds)
            self.compute_multiobj_metrics(
                full_char_loss, char_scores, label_inds, prefix='full'
            )
            if full_char_loss.mean() < self.partial_loss_threshold:
                warn_once('Threshold low enough for partial loss computation')
                self.crossed_partial_loss_threshold = True
            if full_char_loss.mean() < self.sliced_loss_threshold:
                warn_once('Threshold low enough for sliced loss computation')
                self.crossed_sliced_loss_threshold = True
        if (
            'sliced' in self.multiobjective_losses
            and self.crossed_sliced_loss_threshold
        ):
            slice_tok = random.randint(1, batch.label_vec.size(1))
            char_scores = self.get_multiobjective_output(
                latent[:, 0:slice_tok, :], encoder_states, cand_vecs, 'sliced'
            )
            sliced_char_loss = self.multiobj_criterion(char_scores, label_inds)
            self.compute_multiobj_metrics(
                sliced_char_loss, char_scores, label_inds, prefix='sliced'
            )
        if (
            'partial' in self.multiobjective_losses
            and self.crossed_partial_loss_threshold
            and batch.batchsize != 1
        ):
            partial_char_loss = self.compute_partial_decoded_loss(
                batch, latent, encoder_states, cand_vecs, label_inds
            )
        char_loss = (
            full_char_loss.mean() + partial_char_loss.mean() + sliced_char_loss.mean()
        )
        ratio = self.opt['multiobjective_loss_ratio']
        if ratio == 1:
            final_loss = char_loss
            loss.detach_()
        elif ratio == 0:
            final_loss = loss
            char_loss.detach_()
        else:
            final_loss = ratio * char_loss + (1 - ratio) * loss

        if return_output:
            return final_loss, model_output
        else:
            return final_loss

    def compute_multiobj_metrics(
        self,
        char_loss: torch.Tensor,
        scores: torch.Tensor,
        label_inds: torch.LongTensor,
        prefix: str = '',
    ):
        """
        Compute multi-objective metrics to track performance..

        :param char_loss:
            character loss (non-averaged) for each batch item
        :param scores:
            scores for character candidates
        :param label_inds:
            indices of correct characters
        """
        prefix = f'{prefix}_' if prefix else ''
        batchsize = scores.size(0)
        _, ranks = scores.topk(1, 1, largest=True)
        ranks_m = []
        mrrs_m = []
        hits_m = []
        for b in range(batchsize):
            rank = (ranks[b] == label_inds[b]).nonzero()
            rank = rank.item() if len(rank) == 1 else (scores.size(1) - 1)
            ranks_m.append(1 + rank)
            mrrs_m.append(1.0 / (1 + rank))
            hits_m.append(int(rank == 0))
        self.record_local_metric(f'{prefix}rank', AverageMetric.many(ranks_m))
        self.record_local_metric(f'{prefix}hits@1', AverageMetric.many(hits_m))
        self.record_local_metric(f'{prefix}mrr', AverageMetric.many(mrrs_m))
        self.record_local_metric(
            f'{prefix}mean_character_loss', AverageMetric.many(char_loss)
        )

    ##########################################
    # Override non-interesting TRA Functions #
    ##########################################
    def score_candidates(self, *args, **kwargs):
        pass

    def rank_eval_label_candidates(self, *args, **kwargs):
        return None, None


class LongMultiObjectiveGeneratorAgent(MultiObjectiveGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        MultiObjectiveGeneratorAgent.add_cmdline_args(parser, partial_opt)
        TransformerVariantAgent.add_cmdline_args(parser, partial_opt)
        return parser

    def build_model(self, states=None):
        wrapped_class = TransformerGeneratorReturnLatentModel.with_components(
            encoder=ShiftInvariantEncoder
        )
        model = wrapped_class(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return GenerativeMultiObjectiveModule(model, self.opt, self.dict, self.NULL_IDX)
