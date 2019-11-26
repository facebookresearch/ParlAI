#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
DrQA model (reader only) for SQuAD.
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    fnames = ['squad_fasttextcc.tgz']
    opt['model_type'] = 'squad'  # for builder
    download_models(opt, fnames, 'drqa', use_model_type=True, version='v2.0')
