# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Interact with a pre-trained model.
Key-Value Memory Net model trained on personachat using persona 'self'
[Note: no persona in this example code is actually given to the model.]
"""

from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive import interactive

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_params(
        model='projects.personachat.kvmemnn.kvmemnn:KvmemnnAgent',
        model_file='models:convai2/kvmemnn/model',
        interactive_mode=True,
    )
    opt = parser.parse_args()
    # build all profile memory models
    fnames = ['kvmemnn.tgz']
    opt['model_type'] = 'kvmemnn'  # for builder
    download_models(opt, fnames, 'convai2')
    interactive(opt)
