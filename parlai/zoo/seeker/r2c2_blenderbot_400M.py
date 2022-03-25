#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
R2C2 400M Model, fine-tuned on BST Tasks.
"""
from parlai.zoo.seeker.seeker_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'r2c2_blenderbot_400M', 'v1.0')
