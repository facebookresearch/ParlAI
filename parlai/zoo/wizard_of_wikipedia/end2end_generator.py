#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained wizard of wikipedia end2end generative model.
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    fnames = ['end2end_generator_0.tar.gz']
    download_models(
        opt, fnames, 'wizard_of_wikipedia', version='v0.5', use_model_type=False
    )
