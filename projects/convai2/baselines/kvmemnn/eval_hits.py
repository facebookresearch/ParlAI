# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for hits@1 metric
Key-Value Memory Net model trained on convai2:self
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_hits import setup_args, eval_hits

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:convai2/kvmemnn/model',
        numthreads=40,
    )
    opt = parser.parse_args(print_args=False)
    # build all profile memory models
    fnames = ['kvmemnn.tgz']
    opt['model_type'] = 'kvmemnn' # for builder
    download_models(opt, fnames, 'convai2')
    eval_hits(opt, print_parser=parser)
