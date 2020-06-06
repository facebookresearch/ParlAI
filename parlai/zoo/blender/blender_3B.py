#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
2.7B parameter Blender model: please see <parl.ai/project/blender>.
"""

from .build import build

VERSION = 'v1.0'


def download(datapath):
    build(datapath, 'BST3B.tgz', model_type='blender_3B', version=VERSION)
