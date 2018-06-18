# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for f1 metric."""

from parlai.core.build_data import download_models
from parlai.scripts.eval_model import eval_model, setup_args


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        task='twitter',
        datatype='valid',
        metrics='f1',
        model='seq2seq',
        model_file='models:twitter/seq2seq/twitter_seq2seq_model',
        dict_lower=True,
        batchsize=32,
    )
    opt = parser.parse_args(print_args=False)
    if 'twitter/seq2seq/twitter_seq2seq_model' in opt.get('model_file', ''):
        opt['model_type'] = 'seq2seq'
        fnames = ['twitter_seq2seq_model.tgz']
        download_models(opt, fnames, 'twitter', version='v1.0', use_model_type=True)

    eval_model(opt, print_parser=parser)
