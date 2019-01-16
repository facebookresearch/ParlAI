#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate pre-trained model trained for ppl metric.
This language model model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='language_model',
        model_file='models:convai2/language_model/model',
        dict_file='models:convai2/language_model/model.dict',
        batchsize=20,
    )
    opt = parser.parse_args()
    opt['model_type'] = 'language_model'
    fnames = ['model', 'model.dict', 'model.opt']
    download_models(opt, fnames, 'convai2', version='v2.0',
                    use_model_type=True)
    eval_f1(opt, print_parser=parser)
