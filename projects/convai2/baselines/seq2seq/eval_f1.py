#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate pre-trained model trained for f1 metric.
This seq2seq model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='legacy:seq2seq:0',
        model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        dict_file='models:convai2/seq2seq/convai2_self_seq2seq_model.dict',
        dict_lower=True,
        batchsize=32,
    )
    opt = parser.parse_args(print_args=False)
    if (opt.get('model_file', '')
            .find('convai2/seq2seq/convai2_self_seq2seq_model') != -1):
        opt['model_type'] = 'seq2seq'
        fnames = ['convai2_self_seq2seq_model.tgz',
                  'convai2_self_seq2seq_model.dict',
                  'convai2_self_seq2seq_model.opt']
        download_models(opt, fnames, 'convai2', version='v3.0')
    eval_f1(opt, print_parser=parser)
