#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained BERT Classifer model on the multi-turn adversarial dialogue safety task.
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}  # for builder
    fnames = ['safety_models_v1.tgz']
    download_models(
        opt, fnames, 'dialogue_safety', version='v0.5', use_model_type=False
    )
