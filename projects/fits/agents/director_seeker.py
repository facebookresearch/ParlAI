#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import copy

from typing import Optional, List, Tuple, Any, Dict, Union
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.core.message import Message
from parlai.agents.rag.retrievers import Document
from parlai.agents.fid.fid import (
    SearchQueryFiDAgent,
    WizIntGoldDocRetrieverFiDAgent,
)
import parlai.utils.logging as logging
from projects.seeker.agents.seeker_modules import ComboFidModel
from projects.seeker.agents.seeker import (
    ComboFidAgent,
    SeekerAgent,
)
from projects.fits.agents.director_bb2 import (
    DirectorFidAgent,
    DirectorFidModelMixin,
)


class DirectorComboFidModel(DirectorFidModelMixin, ComboFidModel):
    def encoder(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
        skip_retrieval_vec: torch.BoolTensor,
        skip_director_reranking_vec: Optional[torch.BoolTensor] = None,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        Optional[List[List[Document]]],
        Optional[torch.Tensor],
        Optional[torch.BoolTensor],
    ]:
        """
        just pass through the skip_director_reranking_vec.

        :param input:
            2D [bsz, seqlen] input to the encoder
        :param input_lengths:
            1D [bsz] lengths of each input item
        :param query_vec:
            2D [bsz*n_turns, seqlen] input for the retriever
        :param input_turns_cnt:
            1D [bsz] number of dialogue turns for each input example
        :param skip_retrieval_vec:
            1D [bsz] indicator for whether to skip retrieval.
        :param skip_director_reranking_vec:
            1D [bsz] indicator for whether to skip modifying generator scores.
        """
        (
            new_out,
            new_mask,
            input_turns_cnt,
            new_top_docs,
            new_top_doc_scores,
        ) = super().encoder(
            input=input,
            input_lengths=input_lengths,
            query_vec=query_vec,
            input_turns_cnt=input_turns_cnt,
            skip_retrieval_vec=skip_retrieval_vec,
            positions=positions,
            segments=segments,
        )
        return (
            new_out,
            new_mask,
            input_turns_cnt,
            new_top_docs,
            new_top_doc_scores,
            skip_director_reranking_vec,
        )

    def reorder_encoder_states(
        self, encoder_states: Tuple[torch.Tensor, ...], indices: torch.LongTensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, List[List[Document]], torch.Tensor, torch.BoolTensor
    ]:
        """
        Reorder all RAG encoder states.

        pass the last encoder states
        """
        new_enc, new_mask = super().reorder_encoder_states(encoder_states[:-1], indices)
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(new_enc.device)
        return new_enc, new_mask, encoder_states[-1][indices]

    def decode_forced(
        self, encoder_states: Tuple[torch.Tensor, ...], ys: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor, Tuple[Any, ...], torch.BoolTensor]:
        assert encoder_states[-1] is None and len(encoder_states) == 6
        return super().decode_forced(encoder_states[:-1], ys)

    def decoder(
        self,
        input: torch.LongTensor,
        encoder_states: Tuple[Any, ...],
        incr_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Refactored Decode, RAG-Style to split the FID.decoder into
        decoder_output_before_final_projection + decoder_output only for generation
        purpose Decode, RAG-Style.

        :param input:
            input for the decoder
        :param encoder_states:
            RAG encoder states
        :param incr_state:
            incremental decoder state

        :return (output, new_incr_state):
            return the output token distribution, as well as new incremental state.
        """
        dec_out, incr_state = self.decoder_output_before_final_projection(
            input, encoder_states[:-1]
        )
        skip_director_reranking_vec = encoder_states[-1]
        dec_out = self.decoder_output(
            dec_out, skip_director_reranking_vec, encoder_states=encoder_states
        )
        return dec_out, incr_state

    def decoder_output(
        self,
        latent: torch.Tensor,
        skip_director_reranking_vec: torch.BoolTensor,
        encoder_states: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Overriding decoder_output method to use classifier heads to modify the generator
        logprobs. This modification allows model to incorporate attribute information
        from classifier while selecting the next tokens.

        Notice that the skip_director_reranking_for_all_generations is moved to Agent config as all subagents share the same model, including model weights and model configs
        Args:
            latent (torch.Tensor): decoder outputs
            skip_director_reranking_vec (torch.BoolTensor): vectors of whether to skip the modification or not
            encoder_states (Tuple[Any]): encoder states

        Returns:
            Modified logprobs.
        """
        classifier_outputs = F.logsigmoid(
            self.classifier_output(latent, encoder_states=encoder_states)
        )
        log_predictor_scores = F.log_softmax(self.generator_output(latent), dim=-1)

        if self.use_one_plus_gamma_variant:
            scores = log_predictor_scores + self.infer_gamma * classifier_outputs
        else:
            scores = (
                2.0 * (1.0 - self.infer_mixing_weights) * log_predictor_scores
                + 2.0 * (self.infer_mixing_weights) * classifier_outputs
            )
        scores[skip_director_reranking_vec, :, :] = log_predictor_scores[
            skip_director_reranking_vec, :, :
        ]
        return F.log_softmax(scores, dim=-1)


class DirectorComboFidAgent(DirectorFidAgent, ComboFidAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        ComboFidAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('DirectorComboFidAgent Group')
        group.add_argument(
            '--skip-director-reranking-key',
            type=str,
            default='skip_director_reranking',
            help='key in observation determining whether to skip director reranking.',
        )
        group.add_argument(
            '--skip-director-reranking-for-all-generations',
            type=bool,
            default=False,
            help='whether to skip reranking for all generations of the model or not',
        )
        return parser

    def build_model(self) -> DirectorComboFidModel:
        if self.generation_model == 't5':
            RuntimeError('T5 currently not supported')
        else:
            model = DirectorComboFidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def _encoder_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.BoolTensor,
        Union[torch.BoolTensor, None],
    ]:
        """
        Override ComboFidAgent._encoder_input to pass through
        skip_director_reranking_vec.
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.skip_retrieval_vec,
            batch.skip_director_reranking_vec,
        )

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        batch = super().batchify(obs_batch, sort)
        valid_exs = [ex for ex in obs_batch if self.is_valid(ex)]
        batch.skip_director_reranking_vec = None
        if valid_exs:
            skip_director_reranking = [
                obs_batch[i].get(self.opt['skip_director_reranking_key'], False)
                for i in batch.valid_indices
            ]
            if self.opt['skip_director_reranking_for_all_generations']:
                skip_director_reranking = [True for _ in skip_director_reranking]
            batch.skip_director_reranking_vec = torch.BoolTensor(
                skip_director_reranking
            )
        return batch


class DirectorComboFidGoldDocumentAgent(
    DirectorComboFidAgent, WizIntGoldDocRetrieverFiDAgent
):
    pass


class DirectorComboFidSearchQueryAgent(DirectorComboFidAgent, SearchQueryFiDAgent):
    pass


class DirectorSeekerAgent(SeekerAgent):
    @classmethod
    def get_additional_agent_args(cls) -> ParlaiParser:
        """
        Return a parser with arguments sourced from several sub models.
        """
        additional_agent_parser = super().get_additional_agent_args()
        DirectorComboFidAgent.add_cmdline_args(additional_agent_parser)
        return additional_agent_parser

    @classmethod
    def add_additional_subagent_args(cls, agent: ParlaiParser) -> ParlaiParser:
        """
        Override seeker.add_additional_args to add default values.
        """
        additional_agent_parser = cls.get_additional_agent_args()
        for action in additional_agent_parser._actions:
            key = max(action.option_strings, key=lambda x: len(x))
            type = action.type

            for prefix in ['krm', 'drm', 'sqm', 'sdm']:
                # ADD Default
                agent.add_argument(
                    f'--{prefix}-{key.strip("-")}',
                    type=type,
                    required=False,
                    default=action.default,
                )
        return agent

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add some cmd line args for kd k2r combo agent.
        """
        super().add_cmdline_args(parser, partial_opt)
        cls.add_additional_subagent_args(parser)
        group = parser.add_argument_group('DirectorSeekerAgent Group')
        group.add_argument(
            '--subagents-to-rerank',
            type=str,
            default='drm',
            help='subagents to apply reranking, concatenated by ",". For example "drm,krm" meaning reranking dialogue and knowledge response generation subagents. By default only dialogue responses are reranked',
        )
        parser.set_defaults(
            krm_skip_director_reranking_for_all_generations=True,
            drm_skip_director_reranking_for_all_generations=False,
            sqm_skip_director_reranking_for_all_generations=True,
            sdm_skip_director_reranking_for_all_generations=True,
        )
        return parser

    def _get_opt_value(self, opt, key, subagent_name):
        subagent_key = f'{subagent_name}_{key}'
        if 'override' in opt:
            if subagent_key in opt['override']:
                return opt['override'][subagent_key]
            if key in opt['override']:
                return opt['override'][key]
        if subagent_key in opt:
            return opt[subagent_key]
        return opt.get(key)

    def __init__(self, opt, shared=None):
        opt_cp = copy.deepcopy(opt)
        self.subagents_to_rerank = opt['subagents_to_rerank'].split(',')
        director_seeker_args = ['skip_director_reranking_for_all_generations']
        for prefix in ['krm', 'drm', 'sqm', 'sdm']:
            # ADD Default
            opt_cp[
                prefix + '_model'
            ] = 'projects.fits.agents.director_seeker:DirectorComboFidSearchQueryAgent'
            opt_cp['override'][prefix + '_model'] = opt_cp[prefix + '_model']

            for opt_key in director_seeker_args:
                # add it to override so that the subagent acknowledges it (by defaults it loads from opt.file and from cmdline)
                subagent_opt_key = prefix + '_' + opt_key
                if subagent_opt_key not in opt_cp['override']:
                    opt_cp['override'][subagent_opt_key] = self._get_opt_value(
                        opt_cp, opt_key, prefix
                    )

            subagent_opt_key = prefix + '_skip_director_reranking_for_all_generations'
            if prefix in self.subagents_to_rerank:
                opt_cp['override'][subagent_opt_key] = False

        # set infer_mixing_weights/infer_gamma for all subagents not just drm. Actually setting --drm-infer-mixing-weights through cmd line WONT work as the model is simply shared across all subagents
        for opt_key in ['infer_mixing_weights', 'infer_gamma']:
            for subagent in self.subagents_to_rerank:
                subagent_opt_key = subagent + '_' + opt_key
                if subagent_opt_key in opt_cp['override']:
                    opt_cp['override']['krm_' + opt_key] = opt_cp['override'][
                        subagent_opt_key
                    ]
                    logging.warn(
                        f'All {self.subagents_to_rerank} share the same infer reranking args {opt_key}'
                    )

        return super().__init__(opt_cp, shared)
