#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utterance generations model for multi-player light (multilight dataset).
2.7B parameter
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os


_PROJECT_NAME = 'multilight'
_VERSION = 'v1.0'


def download(datapath):
    ddir = os.path.join(get_model_dir(datapath), _PROJECT_NAME)
    model_type = 'utterance'
    if not built(os.path.join(ddir, model_type), _VERSION):
        opt = {'datapath': datapath, 'model_type': model_type}
        fnames = [f'model_{_VERSION}.tgz']
        download_models(
            opt, fnames, _PROJECT_NAME, version=_VERSION, use_model_type=True
        )
