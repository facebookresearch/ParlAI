#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
PACER: Partial And Complete Efficient Re-ranking.

See `PacerTreeSearchMixin.modify_logprobs` for a complete description.
"""
import random
import torch
import torch.nn.functional as F
from typing import Optional, Any, Dict, List
from parlai.agents.rag.retrievers import clean_vec

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import (
    TreeSearch,
    GreedySearch,
    BeamSearch,
    DelayedBeamSearch,
    TopKSampling,
    NucleusSampling,
    TSType,
)
import parlai.utils.logging as logging
from parlai.utils.torch import neginf
from projects.light_whoami.agents.rpa_rerank import (
    RPAReranker,
    RPARerankAgent,
    LongRPARerankAgent,
)
from projects.light_whoami.task.utils import extract_characters
from projects.msc.agents.long_tga import TransformerVariantAgent

from parlai.agents.reranker.reranker import AbstractReranker


class PacerAgentMixin:
    """
    Override TGA to use a different tree search decoder.
    """

    @classmethod
    def get_partial_only_reranker_class(cls) -> AbstractReranker:
        """
        Return class to instantiate classifier.
        """
        return RPAReranker

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        reranker_class = cls.get_partial_only_reranker_class() or AbstractReranker
        reranker_class.add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('PACER Group')
        group.add_argument(
            '--pacer-n-tokens',
            type=int,
            default=10,
            help='How many tokens to re-rank and consider',
        )
        group.add_argument(
            '--pacer-frequency-ratio',
            type=float,
            default=0.05,
            help='The frequency with which to apply PACER re-ranking.',
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        reranker_class = self.get_partial_only_reranker_class()
        if not (shared and 'classifier' in shared):
            self.classifier = reranker_class(opt)
        else:
            self.classifier = shared['classifier']
        assert opt[
            'beam_block_full_context'
        ], 'must set --beam-block-full-context True to use PACER'

    def share(self) -> Dict[str, Any]:
        shared = super().share()
        shared['classifier'] = self.classifier
        return shared

    def _get_batch_context(self, batch):
        """
        Override to always provide full context.
        """
        if 'full_text_vec' not in batch:
            logging.warn('Batch does not have full text vec, resorting to text vec')
            return batch.text_vec
        return batch.full_text_vec

    def _treesearch_factory(self, device: int, verbose=False) -> TreeSearch:
        method = self.opt.get('inference', 'greedy')
        beam_size = self.opt.get('beam_size', 1)
        pacer_kwargs = {
            'classifier': self.classifier,
            'pacer_n_tokens': self.opt['pacer_n_tokens'],
            'pacer_frequency_ratio': self.opt['pacer_frequency_ratio'],
            'agent': self,
        }
        if method == 'greedy':
            return PacerGreedySearch(
                beam_size,
                min_length=0,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                **pacer_kwargs,
            )
        elif method == 'beam':
            return PacerBeamSearch(
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                **pacer_kwargs,
            )
        elif method == 'delayedbeam':
            return PacerDelayedBeamSearch(
                self.opt['topk'],
                self.opt['beam_delay'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                **pacer_kwargs,
            )
        elif method == 'topk':
            return PacerTopKSampling(
                self.opt['topk'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                **pacer_kwargs,
            )
        elif method == 'nucleus':
            return PacerNucleusSampling(
                self.opt['topp'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
                **pacer_kwargs,
            )
        else:
            raise NotImplementedError(
                f'Other gen methods not available for PACER: {method}'
            )


class PacerTreeSearchMixin(TreeSearch):
    def __init__(self, *args, **kwargs):
        self.classifier = kwargs.pop('classifier')
        self.agent = kwargs.pop('agent')
        self.n_toks = kwargs.pop('pacer_n_tokens')
        self.frequency = kwargs.pop('pacer_frequency_ratio')
        super().__init__(*args, **kwargs)

    def get_target_character(self):
        return extract_characters(self.context_str)['_self_name']

    def set_batch_context(
        self: TSType,
        batch_context_list: List[List[int]],
        batch_idx: int,
        gpu_beam_blocking: bool,
    ) -> TSType:
        """
        Override to save de-tokenized version of context.
        """
        # remove pad_idx from the batch vec
        self.context = batch_context_list[batch_idx]
        clean_context = clean_vec(
            batch_context_list[batch_idx], self.agent.END_IDX, [self.agent.NULL_IDX]
        )
        self.context_str = self.agent._v2t(clean_context)
        self.character = self.get_target_character()
        return self

    def select_paths(
        self, logprobs: torch.Tensor, prior_scores: torch.Tensor, current_length: int
    ):
        """
        Override select_paths to modify the logprobs according to classifier outputs.

        :param logprobs:
            a (beamsize x vocab) tensor of log probabilities. If this is the first
            turn in the dialogue, it will be a (1 x vocab) tensor.
        :param prior_scores:
            a (beamsize) tensor of weights with the cumulative running
            log-probability of each beam. If the first turn, it will be a (1) tensor.
        :param current_length:
            the current length in tokens
        :return:
            a (hypothesis_ids, token_id, scores) tuple, where:

            - hypothesis_ids is a LongTensor of hypotheses we're extending. May have
              repeats, but should always be (beamsize) long.
            - token_ids is a (beamsize) LongTensor of next-token choices for
              each of the hypotheses.
            - scores is a (beamsize) Tensor with the updated cumulative log-probs
              of each beam.
        """
        logprobs = self.modify_logprobs(logprobs)
        return super().select_paths(logprobs, prior_scores, current_length)

    def modify_logprobs(self, logprobs: torch.Tensor) -> torch.Tensor:
        """
        Modify logprobs in PACER.

        The way it works:

        1. With frequency r, select a token x_i+1 to re-rank.
        2. Generate word probabilities for token x_i+1.
        3. Examine top k words {x_j | score(x_j) in top_k(P(x_i+1 | x_0,...,x_i))}; use classifier to predict P(a|x1, ..., x_i, x_j)
        4. Rescore top k words via multiplication, re-normalize, and advance the generation.

        :param logprobs:
            initial token probabilities

        :return modified:
            return the modified log probabilities according to PACER
        """
        if random.random() > self.frequency:
            return logprobs
        vals, inds = logprobs.topk(self.n_toks, dim=-1, sorted=False)
        new_probs = logprobs.clone().fill_(neginf(logprobs.dtype))
        # Construct partial hypotheses for each beam for each top K tokens
        batch_hyps = [
            h
            for i in range(len(self.partial_hyps))
            for h in [
                self.agent._v2t(self.partial_hyps[i][1:].tolist() + [ind])
                for ind in inds[i]
            ]
        ]
        # Classify all beam outputs
        predictor_outputs = self.classifier.batch_classify(
            [self.context_str] * self.n_toks * logprobs.size(0), batch_hyps
        )
        # Extract RPA scores
        log_predictor_scores = (
            torch.stack(
                [
                    F.log_softmax(pred['sorted_scores'].float(), dim=0)[
                        pred['text_candidates'].index(self.character)
                    ]
                    for pred in predictor_outputs
                ]
            )
            .to(vals.device)
            .view(vals.size())
        )
        # "Multiply" Probabilities (in log space...)
        scores = vals + log_predictor_scores
        for i in range(new_probs.size(0)):
            new_probs[i, inds[i]] = scores[i]
        return F.log_softmax(new_probs, dim=-1, dtype=torch.float32)  # type: ignore


class PacerGreedySearch(PacerTreeSearchMixin, GreedySearch):
    """
    Override Greedy to work with PACER.
    """

    pass


class PacerBeamSearch(PacerTreeSearchMixin, BeamSearch):
    """
    Override Beam to work with PACER.
    """

    pass


class PacerDelayedBeamSearch(PacerTreeSearchMixin, DelayedBeamSearch):
    """
    Override Delayed Beam Search to work with PACER.
    """

    pass


class PacerTopKSampling(PacerTreeSearchMixin, TopKSampling):
    """
    Override TopK Sampling to work with PACER.
    """

    pass


class PacerNucleusSampling(PacerTreeSearchMixin, NucleusSampling):
    """
    Override Nucleus Sampling to work with PAcer.
    """

    pass


class PacerPartialOnlyAgent(PacerAgentMixin, TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        PacerAgentMixin.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser


class LongPacerPartialOnlyAgent(PacerAgentMixin, TransformerVariantAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerVariantAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        PacerAgentMixin.add_cmdline_args(parser, partial_opt)
        return parser


class PacerAgent(PacerPartialOnlyAgent, RPARerankAgent):
    """
    PACER Agent: Combines Beam and Partial Re-ranking.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        RPARerankAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        PacerPartialOnlyAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser


class LongPacerAgent(LongPacerPartialOnlyAgent, LongRPARerankAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        LongRPARerankAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        LongPacerPartialOnlyAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser
