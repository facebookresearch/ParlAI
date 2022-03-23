#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
LIGHT Am I Me or You Generative-Rerank Agent.

The retriever reranks beam texts via the following architecture setup:

Dialogue Generator:
    Produces the ranked list of beam search candidates.
Character Poly:
    (optionally) Predicts __whoami__ from dialogue history
    Predicts __whoareyou__ from augmented history for each candidate c in C
    Re-ranks beam texts

NOTE: Current assumption is that this is between 2 characters **ONLY** (multiparty support will be added later).
"""
from typing import Optional, List, Dict
from parlai.core.agents import create_agent_from_model_file
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser

from projects.light_whoami.task.agents import SpeakerClassifierTeacher
from projects.light_whoami.task.utils import (
    WHO_AM_I,
    WHO_ARE_YOU,
    extract_characters,
    maybe_annotate,
)

from parlai.agents.reranker.reranker import (
    AbstractReranker,
    AbstractGeneratorRerankAgent,
    LongAbstractGeneratorRerankAgent,
)


class RPAReranker(AbstractReranker):
    """
    RPAReranker subclasses the Reranker.

    This does the following:

    1) Determine listener character from dialogue history
        This is done either via extraction from history or via prediction
    2) Construct augmented dialogue history with text candidates
    3) Predict speaker with augmented dialogue history
    4) Re-rank text candidates according to highest speaker score for (predicted) listener character

    The RPA Re-ranker also provides classification functionality, to just simply
    classify incoming responses.
    """

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        SpeakerClassifierTeacher.add_cmdline_args(parser, partial_opt=partial_opt)
        light_reranker = parser.add_argument_group('RPAReranker args')
        light_reranker.add_argument(
            '--predictor-characters-file',
            type=str,
            default=None,
            help='path to newline-delimited list of characters, if fixed candidates is desired',
        )
        light_reranker.add_argument(
            '--reranker-no-cuda',
            type='bool',
            default=False,
            help='specify to not CUDA-fy it',
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        """
        Initializes RPAReranker.
        """
        super().__init__(opt, shared)

        self._init_attributes(opt)

    def init_predictor(self, opt: Opt, shared=None):
        if not shared:
            override = {
                'return_cand_scores': True,
                'datatype': 'valid',
                'no_cuda': opt['reranker_no_cuda'],
                'interactive_mode': opt.get('interactive_mode', True),
                'ignore_bad_candidates': True,
                'encode_candidate_vecs': True,
                'interactive_candidates': 'inline',
            }  # to not init optim
            if opt.get('predictor_characters_file'):
                override['fixed_candidates_path'] = opt['predictor_characters_file']
            self.predictor = create_agent_from_model_file(
                self.predictor_model_file, opt_overrides=override
            )
        else:
            self.predictor = shared['predictor']

    def _init_attributes(self, opt: Opt):
        """
        Given opt dictionary, initialize relevant attributes for the predictor.

        :param opt:
            options dict
        """
        optfile = f"{self.predictor_model_file}.opt"
        opt_from_file = Opt.load(optfile)
        overrides = opt.get('override')
        opt_from_file.update(overrides)

        assert 'num_utterances' in opt_from_file, opt_from_file
        self.num_utterances = opt_from_file['num_utterances']
        self.annotate_speaker = opt_from_file['annotate_speaker']
        self.speaker_annotation_position = opt_from_file['speaker_annotation_position']
        self.speaker_label_type = opt_from_file['speaker_label_type']
        self.speaker_separator = opt_from_file['speaker_separator']
        self.include_context = opt_from_file['include_light_context']

        if opt_from_file.get('exclude_from_context'):
            self.exclude_from_context = opt_from_file['exclude_from_context'].split(',')
        else:
            self.exclude_from_context = []

    @classmethod
    def get_class_to_rerank_for(
        cls, observation: Message, full_context: str
    ) -> Optional[str]:
        """
        The class from the predictor (classifier) that we want to rerank for.

        For LIGHT, this is the _self_name character.
        """
        characters = extract_characters(full_context)
        self_character = characters.get('_self_name', None)
        return self_character

    def is_context(self, utt: str) -> bool:
        return utt.startswith('_')

    def _predict_character_from_context(
        self, context: str, characters: Dict[str, str], who: str
    ) -> str:
        """
        Given context, predict who the character is.

        :param context:
            dialogue context
        :param characters:
            available characters to choose from
        :param who:
            whether to predict self or partner

        :return whoareyou:
            return predicted self character
        """
        assert (
            not self.annotate_speaker
        ), "if annotate speaker, characters would be in dialogue history"
        control_token = WHO_ARE_YOU if who == 'self' else WHO_AM_I
        utterances = self.get_utterances_from_full_context(
            context, include_context=self.include_context
        )

        if self.num_utterances > 0:
            utterances = utterances[: self.num_utterances]

        utterances.insert(-2, control_token)
        limited_context = self.delimiter.join(utterances)
        label_candidates = extract_characters(context)
        act = self.predict(limited_context, predictor_label_candidates=label_candidates)
        return act['text']

    def augment_context(
        self, full_context: str, candidate: str, include_context: Optional[bool] = True
    ) -> List[str]:
        """
        Given context and candidate, augment the context for predicting whoami.

        :param full_context:
            dialogue context:
        :param candidate:
            candidate response
        :param include_context:
            whether to include the context strings

        :return augmented_context:
            return the augmented context.
        """
        utterances = self.get_utterances_from_full_context(
            full_context, include_context=self.include_context
        )

        if self.num_utterances > 0:
            utterances = utterances[: self.num_utterances]

        characters = extract_characters(full_context)
        self_character = characters.get('_self_name', None)
        partner_character = characters.get('_partner_name', None)
        if not self_character:
            self_character = self._predict_character_from_context(
                full_context, characters, who='self'
            )
        if not partner_character:
            partner_character = self._predict_character_from_context(
                full_context, characters, who='partner'
            )

        light_context = [l for l in utterances if self.is_context(l)]
        dialogue = [l for l in utterances if not self.is_context(l)]
        reversed_dialogue = []
        for i, utt in enumerate(reversed(dialogue)):
            if i % 2 == 0:
                # Even: partner spoke
                reversed_dialogue.append((partner_character, utt))
            else:
                # Odd: you spoke
                reversed_dialogue.append((self_character, utt))
        processed_dialogue = [
            maybe_annotate(
                *d,
                self.annotate_speaker,  # type: ignore
                self.speaker_separator,  # type: ignore
                self.speaker_annotation_position,  # type: ignore
            )
            for d in reversed(reversed_dialogue)
        ]
        if include_context:
            utterances = light_context + processed_dialogue
        else:
            utterances = processed_dialogue

        utterances += (WHO_AM_I, candidate)
        return utterances

    @classmethod
    def get_predictor_label_candidates(
        cls, observation: Message, context: str
    ) -> List[str]:
        """
        Get the list of possible predictor classes.

        In this case, it's not static b/c the characters in each conversation are
        different.
        """
        characters = extract_characters(context)
        self_character = characters.get('_self_name', None)
        partner_character = characters.get('_partner_name', None)
        label_candidates = []
        if self_character is not None:
            label_candidates.append(self_character)
        if partner_character is not None:
            label_candidates.append(partner_character)
        return label_candidates

    def classify(self, context: str, response: str) -> Message:
        """
        Classify response to see what character said it.

        :param context:
            dialogue history
        :param response:
            model response

        :return character:
            return classified character
        """

        # 1) Augment context with response candidates
        augmented_context_utterances = self.augment_context(
            context, response, self.include_context
        )
        augmented_context = self.delimiter.join(augmented_context_utterances)

        # 2) Classify
        # NOTE: context is used below in get_predictor_label_candidates(),
        # not augmented_context because if self.num_utterances is used, it can
        # cut off the context & augmented_context would not have the characters.
        label_candidates = self.get_predictor_label_candidates({}, context)
        predictor_output = self.predict(augmented_context, label_candidates)

        return predictor_output

    def batch_classify(
        self, contexts: List[str], responses: List[str]
    ) -> List[Message]:
        """
        Batch Classify a set of contexts/responses.
        """
        augmented_contexts = [
            self.delimiter.join(
                self.augment_context(context, response, self.include_context)
            )
            for context, response in zip(contexts, responses)
        ]
        label_candidates = self.get_predictor_label_candidates({}, contexts[0])
        predictor_outputs = self.batch_predict(augmented_contexts, label_candidates)
        return predictor_outputs


class RPARerankAgent(AbstractGeneratorRerankAgent):
    """
    Generative Re-rank agent for LIGHT Am I Me or You.
    """

    @classmethod
    def get_reranker_class(cls):
        return RPAReranker


class LongRPARerankAgent(LongAbstractGeneratorRerankAgent):
    @classmethod
    def get_reranker_class(cls):
        return RPAReranker
