#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from dataclasses import dataclass
from typing import Generic, Type, TypeVar


class TComponent(ABC):
    class Manifest:
        pass


T = TypeVar('T', bound=TComponent)


# TODO: Figure out a way to get Manifest class directly from T
# ComponentSpec should stay immutable, since it is used as a
# default param. In python, mutating a param default value changes
# it for all subsequent invocations.
@dataclass(frozen=True)
class ComponentSpec(Generic[T]):
    klass: Type[T]
    manifest: TComponent.Manifest
