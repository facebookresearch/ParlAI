#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from dataclasses import dataclass
from torch import nn
from typing import Generic, Optional, Type, TypeVar


class ModularComponent(nn.Module):
    """
    A transformer component, optionally with swappable subcomponents.

    For an example of how to use this, see parlai/agents/examples/transformer_variant.py
    """

    @dataclass
    class Template:
        """
        Define any swappable components by adding their class as a parameter of this
        object.
        """


MC = TypeVar('MC', bound=ModularComponent)


# ModularComponentBuilder should be immutable, since it is used in default
# function parameter values. In python, mutating a param default value changes
# it for all subsequent invocations.
class ModularComponentBuilder(Generic[MC]):
    """
    When a component has swappable subcomponents, use this object to specify both the
    component type and it's subcomponent types at the same time.
    """

    def __init__(
        self, klass: Type[MC], template: Optional[ModularComponent.Template] = None
    ) -> None:
        self._klass = klass
        self._template = template or klass.Template()

    @property
    def klass(self) -> Type[MC]:
        return self._klass

    @property
    def template(self) -> ModularComponent.Template:
        return self._template

    def build(self, *args, **kwargs) -> MC:
        _kwargs = {'template': self._template}
        _kwargs.update(kwargs)
        return self._klass(*args, **_kwargs)


class StaticComponent(nn.Module):
    """
    This doesn't do anything yet, but it can be used to label components that don't have
    swappable subcomponents.

    By default, anything that isn't a ModularComponent is a StaticComponent.
    """

    pass
