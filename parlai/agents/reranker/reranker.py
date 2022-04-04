#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Re-Ranker Object.

Provided with a predictor model file, the re-ranker provides an API for re-ranking
candidate outputs.
"""
import logging
import torch
from abc import ABC, abstractmethod, abstractclassmethod
from typing import List, Optional, Tuple
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import create_agent_from_model_file, Agent
from parlai.core.build_data import modelzoo_path
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import TorchAgent
from parlai.utils.strings import normalize_reply
from parlai.utils.torch import argsort
from parlai.agents.hugging_face.gpt2 import Gpt2Agent
from projects.msc.agents.long_tga import TransformerVariantAgent

RERANKER_STRATEGIES = ['sum_scores', 'hard_choice', 'reranker_score', 'none']


class AbstractReranker(ABC):
    """
    Re-ranker object which uses a predictor model to rerank candidates given.

    This AbstractReranker can be subclassed to use any classifier or ranking model as
    the predictor.
    """

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None):
        reranker = parser.add_argument_group('AbstractReranker Args')
        reranker.add_argument(
            '--normalize-candidates',
            type=str,
            default=False,
            help='Remove spaces and add capitalization as per ParlAI normalize_reply() function',
        )
        reranker.add_argument(
            '--predictor-model-file',
            type=str,
            default=None,
            help='Path to model whose prediction score will be used to rerank, usually a classifier or ranker',
        )
        reranker.add_argument(
            '--reranker-strategy',
            type=str,
            default='reranker_score',
            help='Which strategy to use when re-ranking response candidates. '
            f"Choices: {','.join(RERANKER_STRATEGIES)}",
        )
        reranker.add_argument(
            '--reranker-delimiter',
            type=str,
            default=None,
            help='delimiter for the reranker',
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        """
        Initializes reranker.
        """
        self.predictor_model_file = modelzoo_path(
            opt['datapath'], opt['predictor_model_file']
        )
        self.reranker_strategy = opt['reranker_strategy']
        self.normalize_candidates = opt['normalize_candidates']
        self.delimiter = opt.get('reranker_delimiter', None)
        if not self.delimiter:
            self.delimiter = opt.get('delimiter', '\n')
        self.include_context = True
        self.include_label_cand_only = False
        self.init_predictor(opt, shared)

    def init_predictor(self, opt: Opt, shared=None):
        """
        Initializes Predictor Module.
        """
        if not shared:
            if not opt.get("predictor_model_file"):
                logging.warn(
                    'Reranker MUST specify predictor_model_file unless subclass __init__() sets up the model in its own way (unusual). Skipping predictor setup!'
                )
            else:
                self.predictor = create_agent_from_model_file(self.predictor_model_file)
        else:
            self.predictor = shared['predictor']

    def share(self):
        shared = {}
        shared['predictor'] = self.predictor
        return shared

    def get_utterances_from_full_context(
        self, full_context: str, include_context: Optional[bool] = True
    ) -> List[str]:
        """
        Return utterances to consider from context string.

        :param full_context:
            full context (or dialog history) to consider
        :param include_context:
            whether or not to include the context utterances (like "your persona:...")

        :return utterances:
            return list of utterances
        """
        split_context = full_context.split(self.delimiter)
        context = [l for l in split_context if self.is_context(l)]
        dialogue = [l for l in split_context if not self.is_context(l)]

        if include_context:
            utterances = context + dialogue
        else:
            utterances = dialogue
        return utterances

    def augment_context(
        self, full_context: str, candidate: str, include_context: bool
    ) -> List[str]:
        """
        Given context and candidate, augment the context with the new candidate.

        This is generally the input for the predictor model.

        :param full_context:
            dialogue context:
        :param candidate:
            candidate response
        :param include_context:
            Whether to include the context utterances (as defined by self.is_context())

        :return augmented_context:
            return the augmented context.
        """
        utterances = self.get_utterances_from_full_context(
            full_context, include_context=include_context
        )

        utterances.append(candidate)
        return utterances

    def predict(self, context: str, predictor_label_candidates: List[str]) -> Message:
        """
        Use predictor to predict given augmented context.

        :param context:
            augmented context with response candidates
        :param predictor_label_candidates:
            optional array of label candidates to pass to the predictor

        :return output:
            return output from ranker act
        """
        assert isinstance(self.predictor, Agent)
        obs = Message({'text': context, 'episode_done': True})
        obs['label_candidates'] = predictor_label_candidates
        self.predictor.observe(obs)
        act = self.predictor.act()
        assert isinstance(act, Message)
        return act

    def batch_predict(
        self, contexts: List[str], predictor_label_candidates: List[str]
    ) -> List[Message]:
        """
        Batch operate on a list of contexts.

        :param contexts:
            list of augmented contexts with response candidates
        :param predictor_label_candidates:
            array of label candidates to pass to the predictor
            (currently has to be the same for all the contexts)
        :return outputs:
            return ranker batch replies
        """
        assert isinstance(self.predictor, TorchAgent)
        observations = [{'text': context, 'episode_done': True} for context in contexts]
        processed_obs = []
        for obs in observations:
            obs['label_candidates'] = predictor_label_candidates
            obs = self.predictor.observe(obs)
            self.predictor.self_observe(obs)
            processed_obs.append(obs)
        batch_replies = self.predictor.batch_act(processed_obs)
        return batch_replies

    def _rerank_candidates(
        self,
        reranker_outputs: List[Message],
        response_cands: List[str],
        response_cand_scores: torch.Tensor,
        rerank_for_class: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Re-rank the response candidates given reranker outputs and a strategy.

        Compute reranking differently according to specified rerank strategy

        :param reranker_outputs:
            outputs from reranker
        :param response_cands:
            list of response candidates
        :param response_cand_scores:
            tensor with scored response candidates from initial model
        :param rerank_for_class:
            The class (in the ML sense) we want to select for

        :return (candidates, indices):
            candidates: reranked list of response candidates.
            indices: list of indices into response_cands corresponding to re-rank order
        """
        if self.reranker_strategy == 'hard_choice':
            predicted_class = [
                (i, c)
                for i, c in enumerate(response_cands)
                if reranker_outputs[i]['text'] == rerank_for_class
            ]
            try:
                predicted_indices, predicted_class = [
                    list(l) for l in zip(*predicted_class)
                ]
            except ValueError:
                # none predicted
                predicted_indices, predicted_class = [], []

            predicted_not_class = [
                (i, c)
                for i, c in enumerate(response_cands)
                if reranker_outputs[i]['text'] != rerank_for_class
            ]
            try:
                predicted_not_indices, predicted_not_class = [
                    list(l) for l in zip(*predicted_not_class)
                ]
            except ValueError:
                predicted_not_indices, predicted_not_class = [], []

            candidates = predicted_class + predicted_not_class
            indices = predicted_indices + predicted_not_indices
        elif self.reranker_strategy == 'sum_scores':
            rerank_scores = [
                o['sorted_scores'][o['text_candidates'].index(rerank_for_class)]
                for o in reranker_outputs
            ]
            scores = [
                rerank_scores[i].item() + response_cand_scores[i].item()
                for i in range(len(rerank_scores))
            ]
            candidates, indices = argsort(
                scores,
                response_cands,  # type: ignore
                list(range(len(response_cands))),  # type: ignore
                descending=True,
            )[:2]
        elif self.reranker_strategy == 'reranker_score':
            rerank_scores = [
                o['sorted_scores'][o['text_candidates'].index(rerank_for_class)].item()
                for o in reranker_outputs
            ]
            candidates, indices = argsort(
                rerank_scores,
                response_cands,  # type: ignore
                list(range(len(response_cands))),  # type: ignore
                descending=True,
            )[:2]
        elif self.reranker_strategy == 'none':
            candidates = response_cands
            indices = list(range(len(response_cands)))

        return candidates, indices  # type: ignore

    def rerank(
        self,
        observation: Message,
        response_cands: List[str],
        response_cand_scores: torch.Tensor,
    ) -> Tuple[List[str], List[int]]:
        """
        Re-rank candidates according to predictor score.

        :param observation:
            Message object that includes the dialogue history
        :param response_cands:
            ranked list of model response candidates
        :param response_cand_scores:
            list of model response candidates' scores

        :return (candidates, indices):
            candidates: a re-ranked list of candidates
            indices: list of indices into response_cands corresponding to re-rank order
        """
        full_context = observation['full_text']

        # 0) Normalize the replies if the opt is passed in
        if self.normalize_candidates:
            response_cands = [normalize_reply(c) for c in response_cands]

        # 1) Augment context with response candidates
        if not self.include_label_cand_only:
            contexts = [
                self.augment_context(
                    full_context, cand, include_context=self.include_context
                )
                for cand in response_cands
            ]
            contexts = [self.delimiter.join(utts) for utts in contexts]
        else:
            # This variant only passes in the label candidates (with no dialogue history whatsoever
            # into the ranker. Can be useful for things like e.g. simple utterance-based safety classifiers.
            contexts = response_cands

        # 2) Predict with augmented context
        label_candidates = self.get_predictor_label_candidates(
            observation, full_context
        )
        reranker_outputs = self.batch_predict(contexts, label_candidates)

        # 3) Rerank
        rerank_for_class = self.get_class_to_rerank_for(observation, full_context)
        reranked_candidates, indices = self._rerank_candidates(
            reranker_outputs,
            response_cands,
            response_cand_scores,
            rerank_for_class=rerank_for_class,
        )

        if self.show_debug_logging(observation):
            debug_str = self._construct_debug_logging(
                observation,
                response_cands,
                response_cand_scores,
                reranker_outputs,
                reranked_candidates,
                rerank_for_class,
            )
            # Need print because even logging.WARN swallowed during eval
            print(debug_str)
        return reranked_candidates, indices

    ###################
    # DEBUG CODE ######
    ###################

    def show_debug_logging(self, observation: Message) -> bool:
        """
        Determine whether to show debug logging.

        Override to perform some logic to determine if this is the case.

        :param observation:
            observation under consideration

        :return show_debug:
            return true if we should show debug logging.
        """
        return False

    def _construct_debug_logging(
        self,
        observation: Message,
        response_cands: List[str],
        response_cand_scores: torch.Tensor,
        reranker_outputs: List[Message],
        reranked_candidates: List[str],
        rerank_for_class: str,
    ) -> str:
        """
        Construct debug logging.
        """
        s = '-------------\nFull Context:\n'
        full_context = observation['full_text']
        utterances = full_context.split(self.delimiter)
        count = 0
        for utt in utterances:
            if self.is_context(utt):
                s += utt
            else:
                s += f'{"Human: " if count % 2 == 0 else "Model: "} {utt}\n'
                count += 1

        s += f'Label was: {observation["eval_labels"]}\n'
        s += 'Before Reranking:\n'
        for i, c in enumerate(response_cands):
            rerank_score = reranker_outputs[i]['sorted_scores'][
                reranker_outputs[i]['text_candidates'].index(rerank_for_class)
            ]
            s += f'c: {c}, beam score: {response_cand_scores[i].item()}, rerank score: {rerank_score}\n'

        s += 'Reranked candidates:\n'
        for c in reranked_candidates:
            s += f'c: {c}\n'
        return s

    ########################
    # Abstract Methods #####
    ########################

    @abstractmethod
    def get_class_to_rerank_for(self, observation: Message, full_context: str) -> str:
        """
        The class from the predictor (classifier) that we will rerank to favor.

        This must correspond to an output from the predictor.
        """

    @abstractmethod
    def is_context(self, utt: str) -> bool:
        """
        Determine if incoming utterance is part of the context.

        :param utt:
            an utterance

        :return is_context:
            returns whether the incoming utterance is part of the context
        """

    @abstractmethod
    def get_predictor_label_candidates(
        self, observation: Message, context: str
    ) -> List[str]:
        """
        Get the list of possible predictor classes.

        Subclasses must override to return possible classes, given a context string.
        """


class AbstractGeneratorRerankAgentMixin:
    """
    Generator Rerank Agent.

    Utilize a classifier/predictor model as re-ranker with a generative agent.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        AbstractReranker.add_cmdline_args(parser)
        parser.set_defaults(skip_generation=False)
        gen_agent = parser.add_argument_group("Generator Rerank Agent")
        gen_agent.add_argument(
            '--inference-strategies',
            type=str,
            default=None,
            help='comma-separated list of inference strategies. '
            'if specified, re-rank over several inference strategies',
        )
        gen_agent.add_argument(
            '--debug-mode',
            type='bool',
            default=False,
            help='specify to enable certain debugging procedures.',
        )
        gen_agent.add_argument(
            '--inference-opt-key',
            type=str,
            default='inference',
            help='specify inference opt key for dialogue response model',
        )

        return parser

    def __init__(self, opt: Opt, shared=None):
        """
        Setup reranker.
        """
        super().__init__(opt, shared)
        reranker_class = self.get_reranker_class()
        self.inference_opt_key = opt.get('inference_opt_key', 'inference')
        self.inference_strategies = (
            opt['inference_strategies'] or opt[self.inference_opt_key]
        ).split(',')
        self.debug_mode = opt.get('debug_mode', False)
        if not shared:
            self.reranker = reranker_class(opt, shared=None)
        else:
            self.reranker = reranker_class(opt, shared=shared['reranker'])

    @abstractclassmethod
    def get_reranker_class(cls) -> AbstractReranker:
        """
        Return class to instantiate re-ranker.
        """

    def set_rerank_strategy(self, strategy: str):
        """
        Set new rerank strategy.
        """
        assert strategy in RERANKER_STRATEGIES
        self.reranker.reranker_strategy = strategy

    def share(self):
        """
        Share model parameters.

        :return: dict of shared class and opt
        """
        shared = super().share()
        shared['reranker'] = self.reranker.share()
        return shared

    def set_decoding_method(self, strategy):
        self.opt[self.inference_opt_key] = strategy

    def get_observations_for_reranker(
        self, observations: List[Message], batch_reply: List[Message]
    ) -> List[Message]:
        return observations

    def batch_act(self, observations: List[Message]) -> List[Message]:
        """
        Batch process a list of observations.

        We call batch act directly on the generator, and then individually process each
        batch reply.
        """
        batch_reply = [Message() for _ in range(len(observations))]
        # 1. get all beam texts to consider
        for strategy in self.inference_strategies:
            self.set_decoding_method(strategy)
            inference_batch_reply = super().batch_act(observations)
            for i, resp in enumerate(inference_batch_reply):
                beam_texts = batch_reply[i].get('beam_texts', [])
                batch_reply[i] = resp  # add metrics, other response items
                new_beam_texts = [(*b, strategy) for b in resp.get('beam_texts', [])]
                batch_reply[i].force_set('beam_texts', beam_texts + new_beam_texts)
        # 2. Rerank
        observations_for_reranker = self.get_observations_for_reranker(
            observations, batch_reply
        )
        for observation, generator_response in zip(
            observations_for_reranker, batch_reply
        ):
            if (
                'beam_texts' not in generator_response
                or not generator_response['beam_texts']
            ):
                logging.warn(
                    f'Generator response had no "beam_texts" field and was: {generator_response}. Skipping reranking. Something could be seriously wrong, but this also occurs when batchsize is not a multiple of number of examples'
                )
                continue
            reranked_candidates, indices = self.reranker.rerank(
                observation,
                [b[0] for b in generator_response['beam_texts']],  # text
                torch.tensor([b[1] for b in generator_response['beam_texts']]),  # score
            )
            if self.debug_mode:
                reranked_candidates = [
                    f"*{generator_response['beam_texts'][i][-1]}* {text}"
                    for i, text in zip(indices, reranked_candidates)
                ]
            generator_response.force_set('text', reranked_candidates[0])
            generator_response.force_set('beam_texts', reranked_candidates)
        return batch_reply


class AbstractGeneratorRerankAgent(
    AbstractGeneratorRerankAgentMixin, TransformerGeneratorAgent, ABC
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        AbstractGeneratorRerankAgentMixin.add_cmdline_args(parser, partial_opt)
        reranker_class = cls.get_reranker_class() or AbstractReranker
        reranker_class.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser


class LongAbstractGeneratorRerankAgent(
    AbstractGeneratorRerankAgentMixin, TransformerVariantAgent, ABC
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerVariantAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        AbstractGeneratorRerankAgentMixin.add_cmdline_args(parser, partial_opt)
        reranker_class = cls.get_reranker_class() or AbstractReranker
        reranker_class.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser


class AbstractGpt2RerankAgent(AbstractGeneratorRerankAgentMixin, Gpt2Agent, ABC):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        Gpt2Agent.add_cmdline_args(parser, partial_opt=partial_opt)
        AbstractGeneratorRerankAgentMixin.add_cmdline_args(parser, partial_opt)
        reranker_class = cls.get_reranker_class() or AbstractReranker
        reranker_class.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser
