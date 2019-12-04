#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
KvMemNN model for ConvAI2 (personachat data).
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    opt['model'] = 'projects.personachat.kvmemnn.kvmemnn:Kvmemnn'
    opt['model_file'] = 'models:convai2/kvmemnn/model'
    opt['model_type'] = 'kvmemnn'  # for builder
    fnames = ['kvmemnn.tgz']
    download_models(opt, fnames, 'convai2')
