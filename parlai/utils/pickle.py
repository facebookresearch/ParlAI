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

from pickle import Unpickler as PythonUnpickler


class Unpickler(PythonUnpickler):
    """
    Custom unpickler to handle moved classes and optional libraries.
    """

    def find_class(self, module, name):
        print(module, name)
        return super().find_class(module, name)


def load(*args, **kwargs):
    return Unpickler(*args, **kwargs).load()
