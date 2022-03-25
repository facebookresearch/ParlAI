#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SeeKeR 3B Dialogue Model.
"""
from parlai.zoo.seeker.seeker_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'seeker_dialogue_3B', 'v1.0')
