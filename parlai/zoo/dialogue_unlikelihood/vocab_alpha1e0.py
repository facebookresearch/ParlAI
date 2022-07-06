#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
90M parameter Blender model: please see <parl.ai/projects/dialogue_unlikelihood>.
"""

from .build import build

VERSION = 'v1.0'


def download(datapath):
    build(
        datapath, 'vocab_alpha1e0_v1.tgz', model_type='vocab_alpha1e0', version=VERSION
    )
