#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""File for miscellaneous utility functions and constants."""


class Message(dict):
    """
    Class for observations and actions.

    Functions like a dict, but breaks when writing to fields that already exist.
    """

    def __setitem__(self, key, val):
        if key in self:
            raise RuntimeError(
                'Message already contains key `{}`. If this was intentional, '
                'please use the function `force_set(key, value)`.'.format(key)
            )
        super().__setitem__(key, val)

    def force_set(self, key, val):
        super().__setitem__(key, val)

    def copy(self):
        return Message(self)
