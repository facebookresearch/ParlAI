#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Classifier Gpt2 Re-Ranker Object.

Provided with a classifier model file, the gpt2 re-ranker provides an API for re-ranking
candidate outputs based on maximizing the probability of a given provided class.
"""
from parlai.agents.reranker.reranker import AbstractGpt2RerankAgent
from parlai.agents.reranker.classifier_reranker import ClassifierReranker


class ClassifierGpt2Reranker(ClassifierReranker):
    pass


class ClassifierGpt2RerankerAgent(AbstractGpt2RerankAgent):
    """
    Generative GPT2 Re-rank agent for adding a ClassifierReranker.
    """

    @classmethod
    def get_reranker_class(cls):
        return ClassifierGpt2Reranker
