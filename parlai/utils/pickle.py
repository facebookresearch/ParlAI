#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

import pickle


class FakeAPEXClass:
    pass


class Unpickler(pickle._Unpickler):  # type: ignore
    """
    Custom unpickler to handle moved classes and optional libraries.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith('apex.'):
                # user doesn't have apex installed. We'll deal with this later.
                return FakeAPEXClass
            else:
                if (
                    module == 'parlai.core.utils' or module == 'parlai.utils.misc'
                ) and name == 'Opt':
                    from parlai.core.opt import Opt

                    return Opt
                if module == 'parlai.core.dict' and name == '_BPEHelper':
                    from parlai.utils.bpe import SubwordBPEHelper as _BPEHelper

                    return _BPEHelper

                raise


def load(*args, **kwargs):
    return Unpickler(*args, **kwargs).load()
