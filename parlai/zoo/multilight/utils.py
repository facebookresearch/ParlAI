#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.core.build_data import built, download_models, get_model_dir

PROJECT_NAME = 'multilight'
VERSION = 'v1.0'


def maybe_download(model_type, datapath):
    ddir = os.path.join(get_model_dir(datapath), PROJECT_NAME)
    if not built(os.path.join(ddir, model_type), VERSION):
        opt = {'datapath': datapath, 'model_type': model_type}
        fnames = ['model.tar.gz']
        download_models(opt, fnames, PROJECT_NAME, version=VERSION, use_model_type=True)
