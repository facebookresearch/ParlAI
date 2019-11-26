#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Evaluate pre-trained model trained for ppl metric.
"""

from parlai.scripts.eval_ppl import eval_ppl, setup_args
from projects.twitter.constants import DICT_FILE_30K


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        task='twitter',
        datatype='valid',
        metrics='ppl',
        model='parlai.agents.legacy_agents.seq2seq.seq2seq_v0:PerplexityEvaluatorAgent',
        model_file='models:twitter/seq2seq/twitter_seq2seq_model',
        dict_lower=True,
        batchsize=1,
        numthreads=60,
        no_cuda=True,
        batchindex=0,
    )
    opt = parser.parse_args()
    eval_ppl(opt, dict_file=DICT_FILE_30K)
