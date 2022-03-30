#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Multiobjective (Vanilla 128), Decoder-Only Outputs.
"""
from parlai.zoo.light_whoami.whoami_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'multiobjective', 'v1.0')
