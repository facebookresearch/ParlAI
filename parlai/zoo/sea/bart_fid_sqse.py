#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART-Large 400m FiD model.

It generates search queries and uses them to retrieve docs from a search engine API.
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    ddir = os.path.join(get_model_dir(datapath), 'sea')
    model_type = 'bart_fid_sqse'
    version = 'v1.0'
    if not built(os.path.join(ddir, model_type), version):
        opt = {'datapath': datapath, 'model_type': model_type}
        fnames = [f'model_{version}.tgz']
        download_models(opt, fnames, 'sea', version=version, use_model_type=True)
