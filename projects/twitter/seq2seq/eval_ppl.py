# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric."""

from parlai.core.build_data import download_models
from parlai.scripts.eval_ppl import eval_ppl, setup_args
from projects.twitter.build_dict import build_dict


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        task='twitter',
        datatype='valid',
        metrics='ppl',
        model='parlai.agents.seq2seq.seq2seq:PerplexityEvaluatorAgent',
        model_file='models:twitter/seq2seq/twitter_seq2seq_model',
        dict_lower=True,
        batchsize=1,
        numthreads=60,
        no_cuda=True,
        batchindex=0,
    )
    opt = parser.parse_args()
    if 'twitter/seq2seq/twitter_seq2seq_model' in opt.get('model_file', ''):
        opt['model_type'] = 'seq2seq'
        fnames = ['twitter_seq2seq_model.tgz']
        download_models(opt, fnames, 'twitter', version='v1.0', use_model_type=True)

    eval_ppl(opt, build_dict)
