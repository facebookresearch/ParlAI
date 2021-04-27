#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Tools for annotating modules with lightweight dependency injection.

Primarily designed to swap out individual modules deep within the transformer class hierarchy.

Usage:

```python
@swappable(component=DefaultClass, ...)
class SomeModel(nn.Module):
    ...
```

Within the model, access the swappable classes like so:

```python
self.swappables.component()
```

When instantiating the model, swap out the component like so:

```python
model = SomeModel.with_components(component=NewCustomClass)()
```
"""
from __future__ import annotations
from dataclasses import dataclass
from torch import nn
from typing import Callable, Generic, Optional, Type, TypeVar, Union


class ModularComponent(nn.Module):
    """
    A transformer component, optionally with swappable subcomponents.

    For an example of how to use this, see parlai/agents/examples/transformer_variant.py
    """

    @dataclass
    class Subcomponents:
        """
        Define any swappable components by adding their class as a parameter of this
        object.
        """

    def __init__(self, *args, components: Optional[Subcomponents] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.swappables = components or type(self).Subcomponents()
        assert (
            type(self.swappables) is not ModularComponent.Subcomponents
        ), "ModularComponent objects must define their own Subcomponents"

    @classmethod
    def with_components(cls, **kwargs) -> ModularComponentWrapper:
        return ModularComponentWrapper(
            klass=cls, components=cls.Subcomponents(**kwargs)
        )


MC = TypeVar('MC', bound=ModularComponent)


def modular_type(klass: Type[ModularComponent]):
    """
    Convenience function for type annotating fields that could be either a specific
    ModularComponent subclass or a ModularComponentWrapper around it.
    """
    if issubclass(klass, ModularComponent):
        return Union[Type[klass], ModularComponentWrapper[klass]]
    return Type[klass]


# ModularComponentBuilder should be immutable, since it is used in default
# function parameter values. In python, mutating a param default value changes
# it for all subsequent invocations.
class ModularComponentWrapper(Generic[MC]):
    """
    When a component has swappable subcomponents, use this object to specify both the
    component type and it's subcomponent types at the same time.
    """

    def __init__(
        self,
        klass: Type[MC],
        components: Optional[ModularComponent.Subcomponents] = None,
    ) -> None:
        self._cls = klass
        self._components = components or klass.Subcomponents()

    def __call__(self, *args, **kwargs) -> MC:
        _kwargs = {'components': self._components}
        _kwargs.update(kwargs)
        return self._cls(*args, **_kwargs)


def swappable(**kwargs) -> Callable[[Type], Type[MC]]:
    """
    Decorator for annotating a class as having swappable subcomponents.

    Usage:

    ```python
    @swappable(component=DefaultClass, ...)
    ```
    """

    def wrap(cls) -> Type:
        return _make_class_swappable(cls, **kwargs)

    return wrap


def _make_class_swappable(cls: Type, **kwargs) -> Type[MC]:
    def _class_dict(new_fields, old_dict=None):
        # Sets up the class namespace, along with type annotations
        old_dict = old_dict or {}
        return {
            **old_dict,
            **new_fields,
            '__annotations__': {
                **old_dict.get('__annotations__', {}),
                **{k: type(v) for k, v in new_fields.items()},
            },
        }

    sub_cls = dataclass(
        type('Subcomponents', (ModularComponent.Subcomponents,), _class_dict(kwargs))
    )
    fields = {'Subcomponents': sub_cls, 'swappables': sub_cls()}
    return type(
        cls.__name__, (cls, ModularComponent), _class_dict(fields, cls.__dict__)
    )


class StaticComponent(nn.Module):
    """
    This doesn't do anything yet, but it can be used to label components that don't have
    swappable subcomponents.

    By default, anything that isn't a ModularComponent is a StaticComponent.
    """

    pass
