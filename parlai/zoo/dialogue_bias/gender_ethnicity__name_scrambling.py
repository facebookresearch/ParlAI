#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BlenderBot3B model tuned with the name-scrambling technique to have lower gender and
race/ethnicity bias.
"""

from parlai.core.build_data import download_models


def download(datapath):
    model_type = 'gender_ethnicity__name_scrambling'
    version = 'v1.0'
    opt = {'datapath': datapath, 'model_type': model_type}
    fnames = [f'{version}.tar.gz']
    download_models(
        opt=opt,
        fnames=fnames,
        model_folder='dialogue_bias',
        version=version,
        use_model_type=True,
    )
