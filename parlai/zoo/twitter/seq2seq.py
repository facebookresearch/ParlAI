#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pretrained seq2seq model on twitter with 30k vocab size."""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {
        'datapath': datapath,
        'model_type': 'seq2seq'
    }
    fnames = ['twitter_seq2seq_model.tgz']
    download_models(opt, fnames, 'twitter', version='v1.0', use_model_type=True)
