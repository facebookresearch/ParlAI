#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from typing import List, Optional, Tuple

from parlai.agents.rag.interfaces import RAG
from parlai.agents.rag.modules import DefaultRagModel, QueryModelType
from parlai.agents.rag.retrievers import (
    DPRRetriever,
    RagDprQueryEncoder,
    RetrieverType,
    TFIDFRetriever,
    TFIDFAndDPRRetriever,
)
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.core.torch_generator_agent import TorchGeneratorAgent


class RagAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add RAG Args.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        rag = parser.add_argument_group('RAG Args')
        rag.add_argument(
            '--rag-retriever-type',
            type=str,
            default=RetrieverType.DPR.value,
            choices=[r.value for r in RetrieverType],
        )
        rag.add_argument(
            '--rag-query-model',
            type=str,
            default=QueryModelType.BERT.value,
            choices=[m.value for m in QueryModelType],
        )
        return parser

    ########################################
    # TorchGeneratorAgent Overrides        #
    ########################################

    ###### 1. Model Inputs ######

    def _model_input(
        self, batch: Batch
    ) -> Tuple[torch.LongTensor, List[int], List[str]]:
        """
        Override TGA._model_input to include raw texts.

        Additionally include "forced" knowledge.

        :return (text_vec, text_lengths, text, forced_knowledge):
            text_vec - tokenized batch
            text_lengths - length of each item in the batch
            text - list of text strings (required for retriever)
        """
        text = get_model_queries(batch, self.query_key)
        return (batch.text_vec, batch.text_lengths, text)

    def _encoder_input(
        self, batch: Batch
    ) -> Tuple[torch.LongTensor, List[int], List[str]]:
        """
        Called directly when generating.

        Override TGA._encoder_input to include "texts" if necessary.

        We do not pass "texts" or "forced_knowledge" when generating RAG-Sequence
        as we retrieve prior to generation.

        :return (text_vec, text_lengths, text):
            Encoder input is the following:
                text_vec - tokenized batch
                text_lengths - length of each item in the batch
                text - list of text strings (required for retriever)
                forced_knowledge - list of knowledge passages to force model to use
        """
        text: Optional[List[str]] = None
        if self.rag_model_type == 'token':
            text = get_model_queries(batch, self.query_key)
        return (batch.text_vec, batch.text_lengths, text)

    ##### 2. Standard TGA Function Overrides #####

    def build_model(self) -> RAG:
        manifest = DefaultRagModel.Manifest()

        retriever_type = RetrieverType(self.opt['rag_retriever_type'])
        if retriever_type is RetrieverType.DPR:
            manifest.retriever = DPRRetriever
        elif retriever_type is RetrieverType.TFIDF:
            manifest.retriever = TFIDFRetriever
        elif retriever_type is RetrieverType.TFIDF_AND_DPR:
            manifest.retriever = TFIDFAndDPRRetriever

        query_model = QueryModelType(self.opt['rag_query_model'])
        if query_model in [QueryModelType.BERT, QueryModelType.BERT_FROM_PARLAI_RAG]:
            manifest.retriever_manifest.query_encoder = RagDprQueryEncoder

        return DefaultRagModel(opt=self.opt, dictionary=self.dict, manifest=manifest)


def get_model_queries(batch: Batch, query_key: str) -> List[str]:
    """
    Extract model queries from a batch.

    :aparam batch:
        batch of observations

    :return queries:
        return a batchsize-length list of queries
    """
    if not batch.observations:
        return ['' for _ in range(len(batch.text_lengths))]

    queries = []
    for a in batch.observations:
        q = a.get(query_key)
        if not q:
            queries.append('')
        elif isinstance(q, str):
            queries.append(q)
        else:
            queries.append(q[0])
    return queries
