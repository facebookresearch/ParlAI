#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar, Union

from parlai.core.opt import Opt


class TComponent(ABC):
    @dataclass
    class Manifest:
        pass


T = TypeVar('T', bound=TComponent)


# TODO: Figure out a way to get Manifest class directly from T
@dataclass
class ComponentSpec(Generic[T]):
    klass: Type[T]
    manifest: TComponent.Manifest


class TransformerAttention(ABC):
    @abstractmethod
    def __init__(self, opt: Opt, **kwargs):
        ...

    @abstractmethod
    def forward(  # type: ignore
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        static_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        ...


class TransformerFFN(ABC):
    @abstractmethod
    def __init__(self, opt: Opt, **kwargs):
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


class TransformerEncoderLayer(TComponent):
    @dataclass
    class Manifest:
        self_attention: Type[TransformerAttention]
        feedforward: Type[TransformerFFN]

    @abstractmethod
    def __init__(self, opt: Opt, manifest: Manifest, **kwargs):
        ...

    @abstractmethod
    def forward(
        self, tensor: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        ...


class TransformerEncoder(TComponent):
    @dataclass
    class Manifest:
        layer: ComponentSpec[TransformerEncoderLayer]

    @abstractmethod
    def __init__(
        self,
        opt,
        manifest: Manifest,
        vocabulary_size,
        embedding=None,
        padding_idx=0,
        reduction_type='mean',
        n_segments=None,
        embeddings_scale=None,
        **kwargs,
    ):
        ...

    @abstractmethod
    def forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        ...


class TransformerDecoderLayer(TComponent):
    @dataclass
    class Manifest:
        self_attention: Type[TransformerAttention]
        encoder_attention: Type[TransformerAttention]
        feedforward: Type[TransformerFFN]

    @abstractmethod
    def __init__(self, opt: Opt, manifest: Manifest, **kwargs):
        ...

    def forward(
        self, x, encoder_output, encoder_mask, incr_state=None, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        ...


class TransformerDecoder(TComponent):
    @dataclass
    class Manifest:
        layer: ComponentSpec[TransformerDecoderLayer]

    @abstractmethod
    def __init__(self, opt: Opt, manifest: Manifest, embedding=None, **kwargs):
        ...

    @abstractmethod
    def forward(
        self, input, encoder_state, incr_state=None, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        ...


class Transformer(TComponent):
    @dataclass
    class Manifest:
        encoder: ComponentSpec[TransformerEncoder]
        decoder: ComponentSpec[TransformerDecoder]

    @abstractmethod
    def __init__(self, opt: Opt, dictionary, manifest: Manifest = None, **kwargs):
        ...

    @abstractmethod
    def forward(
        self, input: torch.LongTensor, ys=None, prev_enc=None, **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Any]:
        ...


class DecoderOnlyTransformerLayer(TComponent):
    @dataclass
    class Manifest:
        self_attention: Type[TransformerAttention]
        feedforward: Type[TransformerFFN]

    @abstractmethod
    def __init__(self, opt: Opt, manifest: Manifest, **kwargs):
        ...

    def forward(self, x, incr_state=None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        ...


class DecoderOnlyTransformer(Transformer):
    @dataclass
    class Manifest:
        layer: ComponentSpec[DecoderOnlyTransformerLayer]
