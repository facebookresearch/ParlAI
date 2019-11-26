#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train model for ppl metric with pre-selected parameters.

These parameters have some variance in their final perplexity, but they were used to
achieve the pre-trained model.
"""

from parlai.scripts.train_model import setup_args, TrainLoop
from projects.twitter.constants import DICT_FILE_30K


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='twitter',
        model='seq2seq',
        model_file='/tmp/twitter_seq2seq_model',
        dict_file=DICT_FILE_30K,
        dict_lower=True,
        datatype='train',
        batchsize=32,
        hiddensize=1024,
        embeddingsize=300,
        attention='none',
        numlayers=3,
        rnn_class='lstm',
        learningrate=1,
        dropout=0.1,
        gradient_clip=0.1,
        lookuptable='enc_dec',
        optimizer='sgd',
        embedding_type='glove',
        momentum=0.9,
        bidirectional=False,
        batch_sort=True,
        validation_every_n_secs=600,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=15,
        log_every_n_secs=1,
        numsoftmax=1,
        truncate=150,
    )
    TrainLoop(parser).train()
