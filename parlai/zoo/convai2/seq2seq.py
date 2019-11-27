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
    opt['model'] = 'legacy:seq2seq:0'
    opt['model_file'] = 'models:convai2/seq2seq/convai2_self_seq2seq_model'
    opt['dict_file'] = 'models:convai2/seq2seq/convai2_self_seq2seq_model.dict'
    opt['dict_lower'] = True
    opt['model_type'] = 'seq2seq'  # for builder
    fnames = [
        'convai2_self_seq2seq_model.tgz',
        'convai2_self_seq2seq_model.dict',
        'convai2_self_seq2seq_model.opt',
    ]
    download_models(opt, fnames, 'convai2', version='v3.0')
