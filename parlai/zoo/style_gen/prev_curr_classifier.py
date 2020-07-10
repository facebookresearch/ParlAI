#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Classifier trained on Image-Chat data with personalities used as labels and the previous
and current utterances used as context.
"""

from parlai.core.build_data import download_models


def download(datapath):
    model_type = 'prev_curr_classifier'
    version = 'v1.0'
    opt = {'datapath': datapath, 'model_type': model_type}
    fnames = [f'{version}.tar.gz']
    download_models(
        opt=opt,
        fnames=fnames,
        model_folder='style_gen',
        version=version,
        use_model_type=True,
    )
