#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utterance generations model for multi-player light (multilight dataset).

2.7B parameter
"""


from parlai.zoo.multilight.utils import maybe_download

MODEL_NAME = 'utterance_3B'


def download(datapath):
    maybe_download(MODEL_NAME, datapath)
