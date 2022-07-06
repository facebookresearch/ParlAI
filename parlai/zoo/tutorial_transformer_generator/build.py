#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained mid-sized generative transformer for tutorial purposes.

Source is the original source model for the DodecaDialog paper
(https://parl.ai/projects/dodecadialogue/), but there have been minor tweaks for the
purposes of the tutorial.
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    model_name = 'tutorial_transformer_generator'
    mdir = os.path.join(get_model_dir(datapath), model_name)
    version = 'v1'
    if not built(mdir, version):
        opt = {'datapath': datapath}
        fnames = ['tutorial_transformer_generator_v1.tar.gz']
        download_models(opt, fnames, model_name, version=version, use_model_type=False)
