#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
The speaker prediction model for MultiLIGHT.
"""


from parlai.zoo.multilight.utils import maybe_download

MODEL_NAME = 'speaker'


def download(datapath):
    maybe_download(MODEL_NAME, datapath)
