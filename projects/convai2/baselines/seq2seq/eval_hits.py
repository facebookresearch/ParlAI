# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric.
This seq2seq model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_hits import setup_args, eval_model
from examples.eval_model import setup_args, eval_model


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='seq2seq',
        model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        dict_file='models:convai2/seq2seq/dict_convai2_self',
        dict_lower=True,
        rank_candidates=True,
        batchsize=64,
    )
    opt = parser.parse_args()
    opt['model_type'] = 'seq2seq'
    fnames = ['convai2_self_seq2seq_model.tgz', 'dict_convai2_self']
    download_models(opt, fnames, 'convai2')
    eval_model(parser, printargs=False)
