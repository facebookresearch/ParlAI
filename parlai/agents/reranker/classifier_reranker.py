#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Classifier Re-Ranker Object.

Provided with a classifier model file, the re-ranker provides an API for re-ranking
candidate outputs based on maximizing the probability of a given provided class.
"""
from typing import Optional, List
from parlai.core.agents import create_agent_from_model_file
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser

from parlai.agents.reranker.reranker import (
    AbstractReranker,
    AbstractGeneratorRerankAgent,
)


class ClassifierReranker(AbstractReranker):
    def __init__(self, opt: Opt, shared=None):
        """
        Initializes reranker.
        """
        super().__init__(opt, shared)
        self.include_label_cand_only = opt['include_label_cand_only']
        self.target_label = opt['target_label']

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        classifier_reranker = parser.add_argument_group('ClassifierReranker args')
        classifier_reranker.add_argument(
            '--include-label-cand-only',
            type='bool',
            default=False,
            help='When passing inputs to the classifier, use only the label targets if set to True.',
        )
        classifier_reranker.add_argument(
            '--target-label',
            type=str,
            default='pos',
            help='The name of the target class/label that we want to maximize the probability of.',
        )
        return parser

    def init_predictor(self, opt: Opt, shared=None):
        if not shared:
            override = {
                'return_cand_scores': True,
                'datatype': 'valid',
                'interactive_mode': opt.get('interactive_mode', True),
                'ignore_bad_candidates': True,
                'encode_candidate_vecs': True,
                'interactive_candidates': 'inline',
            }  # to not init optim
            self.predictor = create_agent_from_model_file(
                self.predictor_model_file, opt_overrides=override
            )
        else:
            self.predictor = shared['predictor']

    def batch_classify(
        self, contexts: List[str], predictor_label_candidates: List[str]
    ) -> Message:
        """
        Use predictor to predict given augmented context.

        :param context:
            augmented context with response candidates
        :param predictor_label_candidates:
            optional array of label candidates to pass to the predictor
        :return output:
            return output from ranker act
        """
        cands = []
        if self.include_label_cand_only:
            for c in predictor_label_candidates:
                cands.append(c)
        else:
            cands = [
                context + self.delimiter + response
                for context, response in zip(contexts, predictor_label_candidates)
            ]
        predictor_outputs = self.batch_predict(cands, self.predictor.class_list)
        return predictor_outputs

    def get_class_to_rerank_for(
        self, observation: Message, full_context: str
    ) -> Optional[str]:
        return self.target_label

    def is_context(self, utt: str) -> bool:
        return False

    def get_predictor_label_candidates(
        self, observation: Message, context: str
    ) -> List[str]:
        return self.predictor.class_list


class ClassifierRerankerAgent(AbstractGeneratorRerankAgent):
    """
    Generative Re-rank agent for adding a ClassifierReranker.
    """

    @classmethod
    def get_reranker_class(cls):
        return ClassifierReranker
