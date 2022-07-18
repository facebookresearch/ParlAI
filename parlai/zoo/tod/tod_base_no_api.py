#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Task-Oriented Dialog pretrained model *without* usage of API Schemas, as described in
(Chen et al 2021); see https://arxiv.org/abs/2110.06905.
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    ddir = os.path.join(get_model_dir(datapath), 'tod_base_no_api')
    model_type = 'tod_base_no_api'
    version = 'v2.0'
    if not built(os.path.join(ddir, model_type), version):
        opt = {'datapath': datapath, 'model_type': model_type}
        fnames = ['model_v2.tar.gz']
        download_models(
            opt, fnames, 'tod', version=version, path='aws', use_model_type=True
        )
