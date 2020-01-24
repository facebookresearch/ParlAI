#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained polyencoder retrieval model on the ConvAI2 dialogue task.
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    version = ['v1.0']
    fnames = [f'convai2_single_task_{version}.tar.gz']
    download_models(opt, fnames, 'blended_skill_talk', version=version)
