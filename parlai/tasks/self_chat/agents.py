#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Model self chat.
"""

from parlai.core.teachers import Teacher


class DefaultTeacher(Teacher):
    def __init__(self, opt, shared=None):
        raise RuntimeError(
            '-t self_chat is a dummy helper, and not meant to be used directly.'
        )
