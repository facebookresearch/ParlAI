#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
FairScore Demographic Perturber model, which is BART finetuned on the PANDA dataset consisting of demographic rewrites.

See paper for more details: https://arxiv.org/abs/2205.12586
"""
from parlai.core.build_data import built, download_models
import os
import os.path


def download(datapath):
    opt = {'datapath': datapath}
    version = 'v1'
    fnames = [f'models_{version}.tar.gz']
    # Path to local build
    local_dir = os.path.join(datapath, 'models', 'fairscore')
    if not built(local_dir, version):
        print("Downloading models to {}".format(local_dir))
        download_models(
            opt,
            fnames,
            model_folder='fairscore',
            version=version,
            use_model_type=False,
        )
    else:
        print("Found existing build at {}".format(local_dir))
