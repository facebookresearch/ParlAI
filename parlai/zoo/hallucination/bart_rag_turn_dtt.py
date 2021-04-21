#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pre-trained DPR Model.
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    ddir = os.path.join(get_model_dir(datapath), 'hallucination')
    model_type = 'bart_rag_turn_dtt'
    version = 'v1.0'
    if not built(os.path.join(ddir, model_type), version):
        opt = {'datapath': datapath, 'model_type': model_type}
        fnames = ['model.tgz']
        download_models(
            opt, fnames, 'hallucination', version=version, use_model_type=True
        )
