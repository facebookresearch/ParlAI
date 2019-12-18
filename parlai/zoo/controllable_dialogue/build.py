#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained models from the "What makes a good conversation?" paper.

See https://parl.ai/projects/controllable_dialogue/.
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    fnames = ['models_v1.tar.gz']
    download_models(
        opt, fnames, 'controllable_dialogue', version='v1.0', use_model_type=False
    )
