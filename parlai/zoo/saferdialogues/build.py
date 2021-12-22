#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Blender 2.7B model fine-tuned on the SaFeRDialogues and BST (without persona) tasks.
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    version = 'v0.1'
    fnames = [f'models_{version}.tar.gz']
    download_models(
        opt,
        fnames,
        model_folder='saferdialogues',
        version=version,
        use_model_type=False,
    )
