#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pretrained Transresnet model on the Personality-Captions task.
"""
from parlai.core.build_data import download_models


def download(datapath):
    """
    Download the model.
    """
    opt = {'datapath': datapath, 'model_type': 'transresnet'}
    fnames = ['transresnet.tgz']
    download_models(opt, fnames, 'personality_captions')
