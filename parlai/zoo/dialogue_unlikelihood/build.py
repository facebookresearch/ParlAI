#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dialogue Unlikelihood models: please see <parl.ai/projects/dialogue_unlikelihood>.
"""

import os
from parlai.core.build_data import download_models, built


def build(datapath, fname, model_type, version):
    opt = {'datapath': datapath}
    opt['model_type'] = model_type
    dpath = os.path.join(datapath, 'models', 'dialogue_unlikelihood', model_type)
    if not built(dpath, version):
        download_models(
            opt, [fname], 'dialogue_unlikelihood', version=version, use_model_type=False
        )
