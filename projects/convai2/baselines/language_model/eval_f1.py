# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric.
This language model model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='language_model',
        model_file='models:convai2/language_model/model',
        dict_file='models:convai2/language_model/model.dict',
        batchsize=20,
    )
    opt = parser.parse_args()
    opt['model_type'] = 'language_model'
    fnames = ['model', 'model.dict']
    download_models(opt, fnames, 'convai2', use_model_type=True)
    eval_f1(opt, print_parser=parser)
