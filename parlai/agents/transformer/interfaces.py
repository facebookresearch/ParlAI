#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
import torch
from typing import Any, Dict, Optional, Protocol, Tuple, Type, Union

from parlai.core.opt import Opt


class TransformerAttention(Protocol):
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


class TransformerFFN(Protocol):
    @abstractmethod
    def __init__(self, opt: Opt, **kwargs):
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


class TransformerEncoderLayer(Protocol):
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


class TransformerEncoder(Protocol):
    @dataclass
    class Manifest:
        layer: Type[TransformerEncoderLayer]
        layer_manifest: TransformerEncoderLayer.Manifest

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


class TransformerDecoderLayer(Protocol):
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


class TransformerDecoder(Protocol):
    @dataclass
    class Manifest:
        layer: Type[TransformerDecoderLayer]
        layer_manifest: TransformerDecoderLayer.Manifest

    @abstractmethod
    def __init__(self, opt: Opt, manifest: Manifest, embedding=None, **kwargs):
        ...

    @abstractmethod
    def forward(
        self, input, encoder_state, incr_state=None, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        ...


class Transformer(Protocol):
    @dataclass
    class Manifest:
        encoder: Type[TransformerEncoder]
        encoder_manifest: TransformerEncoder.Manifest
        decoder: Type[TransformerDecoder]
        decoder_manifest: Type[TransformerDecoderLayer]

    @abstractmethod
    def __init__(self, opt: Opt, dictionary, manifest: Manifest = None, **kwargs):
        ...

    @abstractmethod
    def forward(
        self, input: torch.LongTensor, ys=None, prev_enc=None, **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Any]:
        ...


class DecoderOnlyTransformer(Protocol):
    @dataclass
    class Manifest:
        decoder: Type[TransformerDecoder]
        decoder_manifest: Type[TransformerDecoderLayer]

    @abstractmethod
    def __init__(self, opt: Opt, dictionary, manifest: Manifest = None, **kwargs):
        ...

    @abstractmethod
    def forward(
        self, input: torch.LongTensor, ys=None, prev_enc=None, **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Any]:
        ...
