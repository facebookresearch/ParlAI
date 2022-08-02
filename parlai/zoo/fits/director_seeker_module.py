#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
MSC 3B truncate 1024.
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    ddir = os.path.join(get_model_dir(datapath), 'fits')
    model_type = 'director_seeker_module'
    version = 'v0.1'
    if not built(os.path.join(ddir, model_type), version):
        opt = {'datapath': datapath, 'model_type': model_type}
        fnames = [f'model_{version}.tar.gz']
        download_models(opt, fnames, 'fits', version=version, use_model_type=True)
