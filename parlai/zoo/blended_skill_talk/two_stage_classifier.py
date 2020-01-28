#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained classifier used to distinguish utterances as belonging to the ConvAI2, 
EmpatheticDialogues, or Wizard of Wikipedia datasets.
"""

from parlai.core.build_data import download_models


def download(datapath):
    model_type = 'two_stage_classifier'
    version = 'v1.0'
    opt = {'datapath': datapath}
    fnames = [f'{model_type}_{version}.tar.gz']
    download_models(
        opt=opt,
        fnames=fnames,
        model_folder='blended_skill_talk',
        version=version,
        path='http://localhost:8000/blended_skill_talk',
    )
    # TODO: remove `path` arg
