#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SeeKeR LM XL Model.

GPT2-XL model fine-tuned on SeeKeR data.
"""
from parlai.zoo.seeker.seeker_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'seeker_lm_xl', 'v1.0')
