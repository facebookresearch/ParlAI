#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
KvMemNN model for ConvAI2 (personachat data).
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    opt['model'] = 'language_model'
    opt['model_type'] = 'language_model'  # for builder
    fnames = [
        'model',
        'model.dict',
        'model.opt',
    ]
    download_models(opt, fnames, 'convai2', version='v2.0', use_model_type=True)
