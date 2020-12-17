#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
2.7B parameter Blender model distilled to 1.4B parameters (~2x inference speedup).

Please see <parl.ai/project/blender>.
"""

from .build import build

VERSION = 'v1.1'


def download(datapath):
    build(
        datapath,
        f'BST1Bdistill_{VERSION}.tgz',
        model_type='blender_1Bdistill',
        version=VERSION,
    )
