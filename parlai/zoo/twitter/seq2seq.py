# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Pretrained seq2seq model on twitter with 30k vocab size."""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {
        'datapath': datapath,
        'model_type': 'seq2seq'
    }
    fnames = ['twitter_seq2seq_model.tgz']
    download_models(opt, fnames, 'twitter', version='v1.0', use_model_type=True)
