#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate pre-trained model trained for f1 metric
Key-Value Memory Net model trained on convai2:self
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:convai2/kvmemnn/model',
        numthreads=80,
    )
    opt = parser.parse_args(print_args=False)
    # build all profile memory models
    fnames = ['kvmemnn.tgz']
    opt['model_type'] = 'kvmemnn' # for builder
    download_models(opt, fnames, 'convai2')
    eval_f1(parser, print_parser=parser)
