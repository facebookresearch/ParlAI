#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pretrained models used by unit tests.
"""

import os
from parlai.core.build_data import download_models, built, get_model_dir


def download(datapath):
    opt = {'datapath': datapath}
    model_name = 'unittest'
    mdir = os.path.join(get_model_dir(datapath), model_name)
    version = 'v7.1'
    model_filenames = [
        'seq2seq.tar.gz',
        'transformer_ranker.tar.gz',
        'transformer_generator2.tar.gz',
        'memnn.tar.gz',
        'apex_v1.tar.gz',
        'test_bytelevel_bpe_v2.tar.gz',
        'beam_blocking1.tar.gz',
        'context_blocking1.tar.gz',
        'hred_model_v1.tar.gz',
    ]
    if not built(mdir, version):
        download_models(opt, model_filenames, model_name, version=version)
