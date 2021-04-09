#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC
from dataclasses import dataclass
from typing import Generic, Type, TypeVar


class ModularComponent(ABC):
    """
    A transformer component, optionally with swappable subcomponents.

    For an example of how to use this, see parlai/agents/examples/transformer_variant.py
    """

    class Template:
        """
        Define any swappable components by adding their class as a parameter of this
        object.
        """

        pass


MC = TypeVar('MC', bound=ModularComponent)


# TODO: Figure out a way to get Template class directly from MC
# ModularComponentSpec should stay immutable, since it is used as a
# default param. In python, mutating a param default value changes
# it for all subsequent invocations.
@dataclass(frozen=True)
class ModularComponentSpec(Generic[MC]):
    """
    When a component has swappable subcomponents, use this object to specify both the
    component type and it's subcomponent types at the same time.
    """

    klass: Type[MC]
    template: ModularComponent.Template


class StaticComponent(ABC):
    """
    This doesn't do anything yet, but it can be used to label components that don't have
    swappable subcomponents.

    By default, anything that isn't a ModularComponent is a StaticComponent.
    """

    pass
