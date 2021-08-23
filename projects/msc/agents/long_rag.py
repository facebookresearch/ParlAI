#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.

Original Paper: https://arxiv.org/abs/2005.11401

As used in ParlAI: https://arxiv.org/abs/2104.07567
"""
import torch
import torch.nn
import torch.cuda
from typing import Optional, Union, Type
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.agents.fid.fid import Fid, FidModel, T5FidModel


from parlai.agents.rag.rag import (
    RagAgent,
    T5RagAgent,
    BartRagAgent,
    TransformerGeneratorRagAgent,
    RagModelInterface,
)
from parlai.agents.rag.modules import RagModel, RagEncoder

from .long_tga import TransformerVariantAgent, ShiftInvariantForwardEmbeddingMixin


class ShiftInvariantRagEncoder(ShiftInvariantForwardEmbeddingMixin, RagEncoder):
    """
    ShiftInvariant Encoder for RAG.
    """

    def __init__(
        self,
        opt: Opt,
        dictionary: DictionaryAgent,
        embedding: Optional[torch.nn.Embedding] = None,
        padding_idx: int = 0,
        **kwargs,
    ):
        """
        RagEncoder initialization.

        The Rag Seq2seq encoder is just a regular encoder
        """
        self.n_positions_init = opt.get("n_positions_init", None)
        RagEncoder.__init__(
            self,
            opt=opt,
            dictionary=dictionary,
            embedding=embedding,
            padding_idx=padding_idx,
        )


class LongRagModel(RagModel):
    """
    LongRagAgent.

    The RAG Agent interacts with the RAG model mostly via it's RAG Model interface.
    """

    @classmethod
    def build_encoder(cls, opt: Opt, *args, **kwargs):
        return ShiftInvariantRagEncoder(opt, *args, **kwargs)


class LongFidModel(FidModel):
    """
    LongFidModel.

    The RAG Agent interacts with the RAG model mostly via it's RAG Model interface.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        if hasattr(FidModel, 'add_cmdline_args'):
            FidModel.add_cmdline_args(parser, partial_opt)
        longfid_group = parser.add_argument_group('Long Fid Model Args')
        longfid_group.set_defaults(memory_key='full_text')
        longfid_group.add_argument(
            '--fid-ddp-compatible',
            type=bool,
            default=False,
            help=" whethether to set requires_grad = False for DDP compatibility",
        )
        return parser

    @classmethod
    def build_encoder(cls, opt: Opt, *args, **kwargs):
        return ShiftInvariantRagEncoder(opt, *args, **kwargs)


class TransformerGeneratorVariantRagAgent(
    TransformerVariantAgent, TransformerGeneratorRagAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        TransformerVariantAgent.add_cmdline_args(
            parser, partial_opt=partial_opt
        )  # add transformer args
        return parser

    @staticmethod
    def build_rag_model(opt: Opt, dictionary: DictionaryAgent) -> LongRagModel:
        return LongRagModel(opt, dictionary)


GENERATION_AGENTS = {
    'transformer_variant/generator': TransformerGeneratorVariantRagAgent,
    'transformer/generator': TransformerGeneratorRagAgent,
    'bart': BartRagAgent,
    't5': T5RagAgent,
}


class LongRagAgent(TransformerGeneratorVariantRagAgent, RagAgent):
    """
    LongRagAgent.

    The RAG Agent interacts with the RAG model mostly via it's RAG Model interface.
    """

    _generation_agent: Union[Type[RagAgent], Type[TransformerGeneratorVariantRagAgent]]
    _rag_model_interface: RagModelInterface

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add RAG Args.
        """
        RagAgent.add_cmdline_args(parser, partial_opt=None)
        TransformerGeneratorVariantRagAgent.add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            '--generation-model',
            type=str,
            default='transformer_variant/generator',
            help='which generation model to use',
            choices=[
                'transformer_variant/generator',
                'transformer/generator',
                'bart',
                't5',
            ],
        )
        parser.add_argument(
            '--max-memories', type=int, default=10, help='maximum amount of memories. '
        )
        return parser

    @property
    def generation_model(self) -> str:
        return self._generation_model

    @generation_model.setter
    def generation_model(self, model: str):
        self._generation_model = model
        self._generation_agent = GENERATION_AGENTS[model]


class LongFidAgent(LongRagAgent):
    """
    Fusion in Decoder Agent with Rag Encoder swapped with ShiftInvariantRagEncoder.
    """

    @property
    def rag_model_type(self) -> str:
        return self._rag_model_type

    @rag_model_type.setter
    def rag_model_type(self, model: str):
        self._rag_model_type = model
        self._rag_model_interface = Fid(self.opt, self.NULL_IDX)

    def build_model(self) -> Union[T5FidModel, LongFidModel]:
        if self.generation_model == 't5':
            model = T5FidModel(self.opt, self.dict)
        else:
            model = LongFidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model
