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
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar


C = TypeVar('C', bound=object)


class ModularComponent(Generic[C]):
    @dataclass
    class SwappableSubcomponents:
        """
        Define any swappable subcomponents by adding the class (or a constructor) of the
        components as attributes of this object.

        When using the @swappable decorator, this class is created programmatically.
        """

    @classmethod
    def with_components(cls, **kwargs) -> ModularComponentBuilder[ModularComponent[C]]:
        return ModularComponentBuilder(
            klass=cls, subcomponents=cls.SwappableSubcomponents(**kwargs)
        )

    def __init__(
        self, *args, subcomponents: Optional[SwappableSubcomponents] = None, **kwargs
    ):
        """
        Unpacks the swappable_components, then forwards the call up the MRO chain.
        """
        self.swappables = subcomponents or type(self).SwappableSubcomponents()
        assert (
            type(self.swappables) is not ModularComponent.SwappableSubcomponents
        ), "Modular components must declare their own SwappableSubcomponents"
        super().__init__(*args, **kwargs)


MC = TypeVar('MC', bound=ModularComponent)


class ModularComponentBuilder(Generic[MC]):
    """
    When a component has swappable subcomponents, use this object to specify both the
    component type and it's subcomponent types at the same time.
    """

    def __init__(
        self,
        klass: Type[MC],
        subcomponents: Optional[ModularComponent.SwappableSubcomponents] = None,
    ) -> None:
        self._klass = klass
        self._subcomponents = subcomponents or klass.SwappableSubcomponents()

    def __call__(self, *args, **kwargs) -> MC:
        """
        Forward calls to this instance to __init__ of wrapped class.
        """
        return self._klass(*args, subcomponents=self._subcomponents, **kwargs)

    @property
    def swappables(self) -> Any:
        return self._subcomponents

    def swap_components(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self._subcomponents, name, value)


def swappable(**kwargs) -> Callable[[Type[C]], Type[C]]:
    """
    Decorator that adds swappable subcomponents to a class.

    Usage:

    ```python
    @swappable(component_name=DefaultComponentClass, ...)
    ```
    """

    # Decorators need to return callables that accept only the decorated object.
    # To comply, bundle kwargs into a function that accepts only one argument.
    def wrap(cls: Type[C]) -> Type[C]:
        return _make_class_swappable(cls, **kwargs)

    return wrap


def _make_class_swappable(cls: Type[C], **kwargs) -> Type[C]:
    """
    Creates a new class that subclasses ModularComponent.

    Modifies that class to to accept constructors for the components passed to the
    decorator.
    """

    def _class_dict(new_fields) -> Dict[str, Any]:
        """
        Sets up the class attributes, along with type annotations.
        """
        return {
            **new_fields,
            '__annotations__': {**{k: type(v) for k, v in new_fields.items()}},
        }

    # Create SwappableSubcomponents dataclass with components passed to @swappable
    subcomponent_class_name = ModularComponent.SwappableSubcomponents.__name__
    subcomponent_dataclass = dataclass(
        type(subcomponent_class_name, (), _class_dict(kwargs))
    )

    # Create a new class that subclasses the decorated class. This new class holds
    # the SwappableSubcomponents dataclass created above (list of components that are
    # swappable) and the swappables attribute (the actual swappable component constructors).
    return type(
        # We append "_Swappable" to the new class name for transparency.
        f"{cls.__name__}_Swappable",
        # ModularComponent comes before the class so we can intercept __init__ calls.
        (ModularComponent, cls),  # type: ignore
        # Items in this dictionary are converted to class attributes by type()
        _class_dict({subcomponent_class_name: subcomponent_dataclass}),
    )
