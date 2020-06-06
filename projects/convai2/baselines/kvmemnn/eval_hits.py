#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Evaluate pre-trained model trained for hits@1 metric Key-Value Memory Net model trained
on convai2:self.
"""

import os

from parlai.core.build_data import download_models
from projects.convai2.eval_hits import setup_args, eval_hits


def main(args=None):
    parser = setup_args()
    parser.set_defaults(
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:convai2/kvmemnn/model',
        numthreads=min(40, os.cpu_count()),
    )
    opt = parser.parse_args(args=args, print_args=False)
    # build all profile memory models
    fnames = ['kvmemnn.tgz']
    opt['model_type'] = 'kvmemnn'  # for builder
    download_models(opt, fnames, 'convai2')
    return eval_hits(opt, print_parser=parser)


if __name__ == '__main__':
    main()
