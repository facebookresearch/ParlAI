#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities for registering and loading model wrappers for safety unit and integration
tests.
"""

import projects.safety_bench.model_wrappers
import importlib
import pkgutil
from typing import Callable, Dict, Type


MODEL_WRAPPER_REGISTRY: Dict[str, Type] = {}


def register_model_wrapper(name: str) -> Callable[[Type], Type]:
    """
    Register a model wrapper so that it is available via the CLI.

    >>> @register_model_wrapper("my_model_name")
    ... class MyModelWrapper:
    ...     pass
    """

    def _inner(cls_):
        global MODEL_WRAPPER_REGISTRY
        MODEL_WRAPPER_REGISTRY[name] = cls_
        return cls_

    return _inner


def load_wrapper_module(wrapper_path: str):
    global MODEL_WRAPPER_REGISTRY
    if wrapper_path in MODEL_WRAPPER_REGISTRY:
        return MODEL_WRAPPER_REGISTRY[wrapper_path]

    raise ModuleNotFoundError(f"Could not find wrapper with path: {wrapper_path}")


def setup_wrapper_registry():
    """
    Loads the modules such that @register_model_wrapper hits for all wrappers.
    """
    for module in pkgutil.iter_modules(
        projects.safety_bench.model_wrappers.__path__,
        'projects.safety_bench.model_wrappers.',
    ):
        importlib.import_module(module.name)
