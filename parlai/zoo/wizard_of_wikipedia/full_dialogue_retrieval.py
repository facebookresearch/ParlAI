#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained retrieval model on the Wizard of Wikipedia dialogue task.
"""
from parlai.core.build_data import download_models


def download(opt):
    fnames = ['full_dialogue_retrieval_model.tgz']
    opt['model_type'] = 'full_dialogue_retrieval_model'
    download_models(opt, fnames, 'wizard_of_wikipedia')
