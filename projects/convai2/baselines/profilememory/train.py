# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""

from parlai.scripts.train_model import setup_args, TrainLoop


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='convai2:self',
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit',
        model_file='/tmp/profilememconvai2',
        dict_lower=True,
        dict_include_valid=True,
        dict_maxexs=-1,
        datatype='train',
        batchsize=128,
        encoder='lstm',
        learningrate=0.001,
        numlayers=1,
        hiddensize=1024,
        dropout=0.2,
        attention='general',
        personachat_attnsentlevel=True,
        personachat_sharelt=True,
        personachat_learnreweight=True,
        personachat_reweight='use',
        truncate=100,
        rank_candidates=True,
        validation_every_n_secs=3,
        validation_metric='f1',
        validation_metric_mode='max',
        validation_patience=10,
        log_every_n_secs=10,
    )
    TrainLoop(parser.parse_args()).train()
