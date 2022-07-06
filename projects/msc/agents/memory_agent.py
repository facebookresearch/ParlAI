#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent code for memory agent.

memory agent allows for read/write memory access.

We modify the model input to include "memories" to write to a memory store.
"""
from typing import Optional, Union
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from projects.blenderbot2.agents.sub_modules import KnowledgeAccessMethod
from projects.blenderbot2.agents.modules import (
    T5BlenderBot2RagModel,
    BlenderBot2RagModel,
    T5BlenderBot2FidModel,
    BlenderBot2FidModel,
    BlenderBot2FidModelMixin,
)

import torch
from projects.blenderbot2.agents.blenderbot2 import BlenderBot2RagAgent
from .long_rag import LongRagAgent, LongRagModel, LongFidAgent, LongFidModel


class MemoryRagAgent(BlenderBot2RagAgent):
    """
    Subclass BlenderBot2RagAgent to provide MemoryModel with appropriate inputs
    (specifically, memory vectors).
    """

    ##########################
    # Housekeeping functions #
    ##########################
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add Memory Agent Args.
        """
        BlenderBot2RagAgent.add_cmdline_args(parser, partial_opt)
        mbot_group = parser.add_argument_group('Memory Agent Args')
        mbot_group.set_defaults(
            knowledge_access_method=KnowledgeAccessMethod.MEMORY_ONLY.value
        )
        mbot_group.set_defaults(memory_key='full_text')
        mbot_group.add_argument(
            '--memory-extractor-phrases',
            type=str,
            default=None,
            help="phrases (separated by ,) used to extract memories from dialogue context. "
            "For example, set to 'your persona:' to limit memories to only lines that "
            "contain 'your persona:'",
        )
        mbot_group.add_argument(
            '--retriever-ignore-phrases',
            type=str,
            default=None,
            help='filter input to the retriever such that any utterance containing '
            'the phrases (separated by ,) will not be given as input.',
        )
        mbot_group.add_argument(
            '--memory-delimiter', type=str, default=None, help='memory delimiter'
        )
        return parser

    #########################
    # MBot-specific Changes #
    #########################

    def _set_query_vec(self, observation: Message) -> Message:
        """
        Override RAG.set_query_vec to optionally filter keys.
        """
        query_str = observation[self._query_key]
        if self.opt['retriever_ignore_phrases']:
            retriever_ignore_phrases = self.opt['retriever_ignore_phrases'].split(",")
            for retriever_ignore_phrase in retriever_ignore_phrases:
                query_str = self._filter_text(
                    query_str,
                    retriever_ignore_phrase,
                    delimiter=self.opt['retriever_delimiter'],
                )
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        observation['query_vec'] = model.tokenize_query(query_str)  # type: ignore
        return observation

    def _set_memory_vec(self, observation: Message) -> Message:
        """
        Tokenize the memories for use in read/write memory scoring.

        :param observation:
            observation with input text.

        :return observation:
            return observation with memory vec.
        """
        mem_vecs = None
        method = KnowledgeAccessMethod(self.opt['knowledge_access_method'])
        if method in [
            KnowledgeAccessMethod.ALL,
            KnowledgeAccessMethod.CLASSIFY,
            KnowledgeAccessMethod.MEMORY_ONLY,
        ]:
            memories = observation[self.opt['memory_key']]
            if isinstance(memories, str):
                memories_splits = []
                if self.opt.get('memory_delimiter', None) is not None:
                    # extract memory separated by memory_delimiter up to the last split (which is the current session)
                    memories_splits = memories.split(
                        self.opt.get('memory_delimiter', '\n')
                    )[:-1]
                if len(memories_splits) == 0:
                    memories = [
                        t
                        for tt in memories.split(self.opt.get('delimiter', '\n'))
                        for t in tt.split('\n')
                    ]
                else:
                    memories = memories_splits
            assert isinstance(memories, list)
            if self.opt['memory_extractor_phrases']:
                # extract from context
                memory_extractor_phrases = self.opt['memory_extractor_phrases'].split(
                    ","
                )
                memories = [
                    m
                    for m in memories
                    if any([mem_phrase in m for mem_phrase in memory_extractor_phrases])
                ]
            model = self.model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = self.model.module
            if memories:
                mem_vecs = [model.tokenize_memory(mem) for mem in memories]

        observation['memory_vec'] = mem_vecs
        return observation


class MemoryLongRagModel(BlenderBot2RagModel):
    """
    BlenderBot2RagModel with seq2seq_encoder = ShiftInvariantRagEncoder
    """

    @classmethod
    def build_encoder(cls, opt: Opt, *args, **kwargs):
        return LongRagModel.build_encoder(opt, *args, **kwargs)


class MemoryLongRagAgent(MemoryRagAgent, LongRagAgent):
    """
    Subclass LongRagAgent to provide MemoryRagAgent with appropriate generation_model
    (in particular MemoryLongRagModel that uses the ShiftInvariantEncoder)
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        MemoryRagAgent.add_cmdline_args(parser, partial_opt)
        LongRagAgent.add_cmdline_args(parser, partial_opt)
        return parser

    def build_model(
        self,
    ) -> Union[T5BlenderBot2RagModel, MemoryLongRagModel, BlenderBot2RagModel]:
        """
        Build and return KnowledgeBotRagModel.
        """
        if self.generation_model == 't5':
            model = T5BlenderBot2RagModel(self.opt, self.dict)
        elif self.generation_model == 'transformer_variant/generator':
            model = MemoryLongRagModel(self.opt, self.dict)
        else:
            model = BlenderBot2RagModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


class MemoryLongFidModel(BlenderBot2FidModelMixin, MemoryLongRagModel, LongFidModel):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        if hasattr(MemoryLongRagModel, 'add_cmdline_args'):
            MemoryLongRagModel.add_cmdline_args(parser, partial_opt)
        LongFidModel.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, dictionary, retriever_shared=None):
        super().__init__(opt, dictionary, retriever_shared)
        if opt.get('fid_ddp_compatible', True):
            for param in self.long_term_memory.query_encoder.parameters():
                param.requires_grad = False
            for param in self.long_term_memory.memory_encoder.parameters():
                param.requires_grad = False
            for param in self.retriever.parameters():
                param.requires_grad = False


class MemoryLongFidAgent(LongFidAgent, MemoryRagAgent):
    """
    Subclass MemoryRagAgent to provide MemoryLongFidAgent with appropriate
    generation_model (in particular MemoryLongFidAgent that uses the
    ShiftInvariantEncoder)
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        MemoryRagAgent.add_cmdline_args(parser, partial_opt)
        LongFidAgent.add_cmdline_args(parser, partial_opt)
        MemoryLongFidModel.add_cmdline_args(parser, partial_opt)
        return parser

    def build_model(
        self,
    ) -> Union[T5BlenderBot2FidModel, MemoryLongFidModel, BlenderBot2FidModel]:
        """
        Build and return MemoryLongFidModel.
        """
        if self.generation_model == 't5':
            model = T5BlenderBot2FidModel(self.opt, self.dict)
        elif self.generation_model == 'transformer_variant/generator':
            model = MemoryLongFidModel(self.opt, self.dict)
        else:
            model = BlenderBot2FidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model
