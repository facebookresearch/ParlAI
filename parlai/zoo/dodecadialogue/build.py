#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained models from The Dialogue Dodecathlon.

https://arxiv.org/abs/1911.03768
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    model_name = 'dodecadialogue'
    mdir = os.path.join(get_model_dir(datapath), model_name)
    version = 'v1.0'
    if not built(mdir, version):
        opt = {'datapath': datapath}
        fnames = ['dodecadialogue.tgz']
        download_models(opt, fnames, model_name, version=version, use_model_type=False)
