#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained LIGHT BERT bi-Ranker dialogue model from:
https://arxiv.org/pdf/1903.03094.pdf
"""

from parlai.core.build_data import download_models

import os


def download(datapath):
    opt = {'datapath': datapath}  # for builder
    fnames = ['safety_models_v1.tgz']
    download_models(
        opt,
        fnames,
        'dialogue_safety',
        version='v0.5',
        use_model_type=False
    )


def _path(datapath):
    return os.path.join(
        datapath,
        'models',
        'dialogue_safety',
        'single_turn',
        'model'
    )
