#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC
from dataclasses import dataclass
from typing import Generic, Type, TypeVar


class TComponent(ABC):
    """
    A transformer component, optionally with swappable subcomponents.
    For an example of how to use this, see
    parlai/agents/examples/transformer_variant.py
    """

    class Manifest:
        """
        Define any swappable components by adding
        their class as a parameter of this object.
        """

        pass


T = TypeVar('T', bound=TComponent)


# TODO: Figure out a way to get Manifest class directly from T
# ComponentSpec should stay immutable, since it is used as a
# default param. In python, mutating a param default value changes
# it for all subsequent invocations.
@dataclass(frozen=True)
class ComponentSpec(Generic[T]):
    """
    When a component has swappable subcomponents, use this object to
    specify both at the same time.
    """

    klass: Type[T]
    manifest: TComponent.Manifest
