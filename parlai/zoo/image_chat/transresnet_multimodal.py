#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pretrained Transresnet Multimodal model on the Image-Chat task.
"""
from parlai.core.build_data import download_models


def download(datapath):
    """
    Download the model.
    """
    opt = {'datapath': datapath, 'model_type': 'transresnet_multimodal'}
    fnames = ['transresnet_multimodal.tgz']
    download_models(opt, fnames, 'image_chat')
