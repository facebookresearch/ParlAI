#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generator trained on dialogue datasets (BST, ConvAI2, ED, and WoW), with style labels
(Image-Chat personalities) attached to train examples 75% of the time.
"""

from parlai.core.build_data import download_models


def download(datapath):
    model_type = 'c75_labeled_dialogue_generator'
    # v1.1 vacuumed the model file to be smaller
    version = 'v1.1'
    opt = {'datapath': datapath, 'model_type': model_type}
    fnames = [f'{version}.tar.gz']
    download_models(
        opt=opt,
        fnames=fnames,
        model_folder='style_gen',
        version=version,
        use_model_type=True,
    )
