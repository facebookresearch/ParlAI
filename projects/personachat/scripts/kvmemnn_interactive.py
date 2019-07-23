#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive import interactive

'''Interact with pre-trained model
Key-Value Memory Net model trained on personachat using persona 'self'
[Note: no persona in this example code is actually given to the model.]
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_params(
        task='parlai.agents.local_human.local_human:LocalHumanAgent',
        model='projects.personachat.kvmemnn.kvmemnn:KvmemnnAgent',
        model_file='models:personachat/kvmemnn/kvmemnn/persona-self_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops1_lins0_model',
        interactive_mode=True,
    )
    opt = parser.parse_args()
    # build all profile memory models
    fnames = ['kvmemnn.tgz']
    opt['model_type'] = 'kvmemnn'  # for builder
    download_models(opt, fnames, 'personachat')
    interactive(opt)
