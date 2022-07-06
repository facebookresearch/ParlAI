#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base R2C2 400M Model.

3 Pre-training datasets (pushshift.io Reddit, RoBERTa, & CC)
"""
from parlai.zoo.seeker.seeker_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'r2c2_base_400M', 'v1.0')
