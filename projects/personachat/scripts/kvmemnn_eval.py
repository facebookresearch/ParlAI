#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from parlai.scripts.eval_model import eval_model

'''Evaluate pre-trained model trained for hits@1 metric
Key-Value Memory Net model trained on personachat using persona 'self'
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.set_defaults(
        task='personachat:self',
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:personachat/kvmemnn/kvmemnn/persona-self_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops1_lins0_model',
        datatype='test',
        numthreads=8,
    )
    opt = parser.parse_args()
    # build all profile memory models
    fnames = ['kvmemnn.tgz']
    opt['model_type'] = 'kvmemnn'  # for builder
    download_models(opt, fnames, 'personachat')

    # add additional model args
    opt['interactive_mode'] = False

    eval_model(parser)
