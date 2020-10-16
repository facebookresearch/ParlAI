#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Detectron Models used in
"12-in-1: Multi-Task Vision and Language Representation Learning" (Lu et. al).

See https://github.com/facebookresearch/vilbert-multi-task and
specifically https://github.com/facebookresearch/vilbert-multi-task/tree/master/data
for more details.
"""

import os
from parlai.core.build_data import download_models, built

BASE_PATH = 'https://dl.fbaipublicfiles.com/vilbert-multi-task'
DETECTRON_MODEL_URL = (
    'https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth'
)
DETECTRON_CONFIG_URL = (
    'https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml'
)


def build(datapath):
    opt = {'datapath': datapath}
    dpath = os.path.join(datapath, 'models', 'detectron')
    fnames = ['detectron_model.pth', 'detectron_config.yaml']
    version = '1.0'
    if not built(dpath, version):
        download_models(
            opt,
            fnames,
            'detectron',
            path=BASE_PATH,
            version=version,
            use_model_type=False,
        )
