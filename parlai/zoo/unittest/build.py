#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pretrained models used by unit tests.
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    model_filenames = [
        'seq2seq.tar.gz',
        'transformer_ranker.tar.gz',
        'transformer_generator2.tar.gz',
        'memnn.tar.gz',
        'apex_v1.tar.gz',
        'test_bytelevel_bpe_v2.tar.gz',
    ]
    download_models(opt, model_filenames, 'unittest', version='v5.0')
