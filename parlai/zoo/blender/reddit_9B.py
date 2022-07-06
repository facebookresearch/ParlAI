#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
9.4B parameter Reddit model: please see <parl.ai/project/blender>.
"""

from .build import build

VERSION = 'v1.0'


def download(datapath):
    build(datapath, 'Reddit9B_v0.tgz', model_type='reddit_9B', version=VERSION)
