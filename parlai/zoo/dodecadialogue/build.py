#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained models from The Dialogue Dodecathlon.

The dodecadialogue_v2.tgz file is always downloaded,
and contains 1) a README.md and 2) the dict for all models.

https://arxiv.org/abs/1911.03768
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath, model_name):
    ddir = os.path.join(get_model_dir(datapath), 'dodecadialogue')
    dodeca_version = 'v2.0'
    if not built(ddir, dodeca_version):
        opt = {'datapath': datapath}
        fnames = ['dodecadialogue_v2.tgz']
        download_models(
            opt, fnames, 'dodecadialogue', version=dodeca_version, use_model_type=False
        )
    model_version = 'v1.0'
    mdir = os.path.join(ddir, model_name)
    if not built(mdir, model_version):
        opt = {'datapath': datapath, 'model_type': model_name}
        fnames = [f'{model_name}.tgz']
        download_models(
            opt, fnames, 'dodecadialogue', version=model_version, use_model_type=True
        )
