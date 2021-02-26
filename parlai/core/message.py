#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
File for Message object and associated functions.

The Message object's key function is to prevent users from editing fields in an action
or observation dict unintentionally.
"""

from __future__ import annotations


class Message(dict):
    """
    Class for observations and actions in ParlAI.

    Functions like a dict, but triggers a RuntimeError when calling __setitem__ for a
    key that already exists in the dict.
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
        return type(self)(self)

    @classmethod
    def padding_example(cls) -> Message:
        """
        Create a Message for batch padding.
        """
        return cls({'batch_padding': True, 'episode_done': True})

    def is_padding(self) -> bool:
        """
        Determine if a message is a padding example or not.
        """
        return bool(self.get('batch_padding'))
