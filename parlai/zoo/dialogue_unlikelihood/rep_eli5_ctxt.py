#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Context repetition unlikelihood on ELI5: please see.

<parl.ai/projects/dialogue_unlikelihood>.
"""

from .build import build

VERSION = 'v1.0'


def download(datapath):
    build(datapath, 'rep_eli5_ctxt_v1.tgz', model_type='rep_eli5_ctxt', version=VERSION)
