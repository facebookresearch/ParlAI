#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from dataclasses import dataclass
import torch
from typing import List, Optional, Protocol, Type, Tuple

from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.agents.rag.classes import Document
from parlai.agents.transformer.interfaces import ComponentSpec, TComponent, Transformer


class RagTokenizer(Protocol):
    @abstractmethod
    def __init__(self, dictionary: DictionaryAgent, **kwargs):
        ...

    @abstractmethod
    def get_pad_idx(self) -> int:
        ...

    @abstractmethod
    def get_bos_idx(self) -> int:
        ...

    @abstractmethod
    def get_eos_idx(self) -> int:
        ...

    @abstractmethod
    def encode(self, txt: str, **kwargs) -> List[int]:
        ...


class RagQueryEncoder(Protocol):
    @abstractmethod
    def __init__(self, opt: Opt, **kwargs):
        ...

    @abstractmethod
    def forward(self, query_vecs: torch.LongTensor, **kwargs) -> torch.Tensor:
        ...


class RagRetriever(TComponent):
    @dataclass
    class Manifest:
        tokenizer: Type[RagTokenizer]
        query_encoder: Type[RagQueryEncoder]

    @abstractmethod
    def __init__(self, opt: Opt, manifest: Manifest = None, **kwargs):
        ...

    @abstractmethod
    def retrieve_and_score(
        self, query_text: List[str], query_vectors: torch.Tensor, **kwargs
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        ...


class RAG(TComponent):
    @dataclass
    class Manifest:
        retriever: ComponentSpec[RagRetriever, RagRetriever.Manifest]
        generator: ComponentSpec[Transformer, Transformer.Manifest]

    @abstractmethod
    def __init__(self, opt: Opt, dictionary, manifest: Manifest = None, **kwargs):
        ...

    @abstractmethod
    def forward(
        self,
        input: torch.LongTensor,
        input_lengths: List[int],
        input_text: Optional[List[str]],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        ...
