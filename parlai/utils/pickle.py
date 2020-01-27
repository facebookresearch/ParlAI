#!/usr/bin/env python3

"""
ParlAI's custom unpickler.

As modules move around or are renamed, it old torch model files become invalid,
since they look for modules in all the wrong places. Furthermore, we occassionally
use APEX for performance reasons, but we don't want to outright die if the user
has not installed it.

This module is to handle both of these issues. It is used like this:

>>> import parlai.utils.pickle
>>> state_dict = torch.load(filename, pickle_module=parlai.utils.pickle)
"""

import sys
import pickle
import unittest.mock as mock


class FakeAPEXClass:
    def __init__(self, *args, **kwargs):
        self._fakeattr = "Hi"

    def __getattr__(self, key):
        if key == '_fakeattr':
            return super().__getattr__(key)
        else:
            return self._fakeattr


class Unpickler(pickle._Unpickler):
    """
    Custom unpickler to handle moved classes and optional libraries.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith('apex.'):
                # user doesn't have apex installed. We'll deal with this later.
                return FakeAPEXClass
            else:
                raise


def load(*args, **kwargs):
    return Unpickler(*args, **kwargs).load()
