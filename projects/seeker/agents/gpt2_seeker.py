#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
GPT2 SeeKeR Agent for Language Modeling.
"""
from typing import Optional, Tuple, List, Union
import torch
import torch.nn

from parlai.agents.fid.fid import (
    FidModel,
    FidAgent,
    SearchQuerySearchEngineFiDAgent,
    WizIntGoldDocRetrieverFiDAgent,
)
from parlai.agents.hugging_face.gpt2 import Gpt2Agent
from parlai.agents.rag.rag import BaseGenerationAgentMixin
from parlai.agents.rag.retrievers import Document
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.torch import padded_tensor

from projects.seeker.agents.seeker import SeekerAgent
from projects.seeker.agents.gpt2_seeker_modules import (
    GPT2WithRetrieverModel,
    ComboGPT2Model,
)


class GPT2WithRetrieverAgentBase(Gpt2Agent, BaseGenerationAgentMixin):
    @staticmethod
    def build_rag_model(
        opt: Opt, dictionary: DictionaryAgent
    ) -> "GPT2WithRetrieverModel":
        return GPT2WithRetrieverModel(opt, dictionary)


class GPT2WithRetrieverAgent(FidAgent, GPT2WithRetrieverAgentBase):
    """
    GPT2 with Retriever agent.

    This agent packs in the retrieved documents and input context as one big "prompt" to
    the language model.
    """

    def __init__(self, opt, shared=None):
        opt['generation_model'] = 'gpt2'
        super().__init__(opt, shared)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        FidAgent.add_cmdline_args(parser, partial_opt)
        Gpt2Agent.add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group('GPT2 Retriever Agent')
        group.add_argument(
            '--filter-docs-with-label',
            type='bool',
            default=False,
            help='If true, we make sure that we do not retrieve any docs containing '
            'label string VERBATIM.',
        )
        return parser

    @staticmethod
    def build_rag_model(
        opt: Opt, dictionary: DictionaryAgent
    ) -> "GPT2WithRetrieverModel":
        return GPT2WithRetrieverModel(opt, dictionary)

    def build_model(self) -> FidModel:
        return GPT2WithRetrieverModel(self.opt, self.dict)

    @property
    def generation_model(self) -> str:
        return self._generation_model

    @generation_model.setter
    def generation_model(self, model: str):
        """
        Override to always be GPT2.
        """
        self._generation_model = model
        self._generation_agent = GPT2WithRetrieverAgentBase

    def _pad_tensor(
        self, items: List[Union[List[int], torch.LongTensor]], is_label: bool = False
    ) -> Tuple[torch.LongTensor, List[int]]:
        """
        Override to always set fp16friendly to False and left_pad to True.
        """
        return padded_tensor(
            items, pad_idx=self.NULL_IDX, left_padded=True, fp16friendly=False
        )

    def _model_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        Optional[torch.LongTensor],
    ]:
        """
        Override model input to return target lengths as well.
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.label_vec.ne(self.NULL_IDX).sum(1)
            if batch.label_vec is not None
            else None,
        )

    def _encoder_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        Optional[torch.LongTensor],
    ]:
        """
        During generation, we send the label truncation to the model.
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.text_vec.new(batch.batchsize).fill_(self.label_truncate),
        )

    def eval_step(self, batch: Batch) -> Optional[Output]:
        """
        Override to cache the labels for retrieval.
        """
        if batch.label_vec is not None and self.opt.get('filter_docs_with_label'):
            self.model_api.set_labels(batch.label_vec)
        output = TorchGeneratorAgent.eval_step(self, batch)
        return output


class GPT2WithRetrieverSearchQueryAgent(
    GPT2WithRetrieverAgent, SearchQuerySearchEngineFiDAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        GPT2WithRetrieverAgent.add_cmdline_args(parser, partial_opt)
        SearchQuerySearchEngineFiDAgent.add_cmdline_args(parser, partial_opt)
        return parser


class GPT2WithRetrieverGoldDocumentAgent(
    GPT2WithRetrieverAgent, WizIntGoldDocRetrieverFiDAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        GPT2WithRetrieverAgent.add_cmdline_args(parser, partial_opt)
        WizIntGoldDocRetrieverFiDAgent.add_cmdline_args(parser, partial_opt)
        return parser


####################
# Combo Agent Code #
####################
class GPT2ComboAgent(GPT2WithRetrieverAgent):
    """
    Combo GPT2 Agent.

    This agent can handle retrieval for some contexts, and vanilla decoder-only
    computation for others.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add Combo Args.
        """
        super().add_cmdline_args(parser, partial_opt)
        gpt2_combo = parser.add_argument_group('GPT2 Combo Group')
        gpt2_combo.add_argument(
            '--skip-retrieval-key',
            type=str,
            default='skip_retrieval',
            help='key in observation determining whether to skip retrieval.',
        )

    def build_model(self) -> ComboGPT2Model:
        """
        Build and return ComboGPT2Model.
        """
        model = ComboGPT2Model(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Overrides batchify to add skip retrieval input vec.
        """
        batch = super().batchify(obs_batch, sort)
        valid_exs = [ex for ex in obs_batch if self.is_valid(ex)]
        if valid_exs:
            skip_retrieval = [
                ex.get(self.opt['skip_retrieval_key'], False) for ex in valid_exs
            ]
            batch.skip_retrieval_vec = torch.BoolTensor(skip_retrieval)
        return batch

    def _model_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
    ]:
        """
        Override _model_input to add skip_retrieval_vec.
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.skip_retrieval_vec,
            batch.label_vec.ne(self.NULL_IDX).sum(1)
            if batch.label_vec is not None
            else None,
        )

    def _encoder_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
    ]:
        """
        For the encoder, during generation, we don't send label vec but rather the label
        truncation.
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.skip_retrieval_vec,
            batch.text_vec.new(batch.batchsize).fill_(self.label_truncate),
        )

    def get_retrieved_knowledge(self, message: Message) -> List[Document]:
        if message.get('skip_retrieval'):
            return []
        return super().get_retrieved_knowledge(message)

    def eval_step(self, batch: Batch) -> Optional[Output]:
        """
        Override to potentially filter docs with the label.

        Additionally add top docs to the output.
        """
        if batch.label_vec is not None and self.opt.get('filter_docs_with_label'):
            self.model_api.set_labels(batch.label_vec)
        output = TorchGeneratorAgent.eval_step(self, batch)
        if output is not None:
            output.top_docs = self.model_api.get_top_docs()
        return output


class GPT2ComboSearchQueryAgent(GPT2ComboAgent, SearchQuerySearchEngineFiDAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        GPT2ComboAgent.add_cmdline_args(parser, partial_opt)
        SearchQuerySearchEngineFiDAgent.add_cmdline_args(parser, partial_opt)
        return parser


class GPT2ComboGpt2GoldDocumentAgent(GPT2ComboAgent, WizIntGoldDocRetrieverFiDAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        GPT2ComboAgent.add_cmdline_args(parser, partial_opt)
        WizIntGoldDocRetrieverFiDAgent.add_cmdline_args(parser, partial_opt)
        return parser


class GPT2SeekerAgent(SeekerAgent):
    @classmethod
    def get_additional_agent_args(cls) -> ParlaiParser:
        """
        Return a parser with arguments sourced from several sub models.
        """
        additional_agent_parser = SeekerAgent.get_additional_agent_args()
        GPT2ComboAgent.add_cmdline_args(additional_agent_parser)
        return additional_agent_parser
