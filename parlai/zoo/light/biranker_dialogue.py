#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained LIGHT BERT bi-Ranker dialogue model from:

https://arxiv.org/pdf/1903.03094.pdf
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath, 'model_type': 'biranker_dialogue'}  # for builder
    fnames = ['biranker_dialogue.tar.gz']
    download_models(opt, fnames, 'light', version='v0.5', use_model_type=True)
