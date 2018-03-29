# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but a run with
these parameters was able to reach 29.54 ppl.
"""

from projects.convai2.baselines.download_models import download
from parlai.core.params import ParlaiParser
from examples.train_model import setup_args, TrainLoop
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='convai2:self',
        model='seq2seq',
        model_file='/tmp/convai2_self_seq2seq_model',
        dict_file='/tmp/dict_convai2_self',
        dict_lower=True,
        dict_maxexs=-1,
        datatype='train',
        batchsize=128,
        hiddensize=1024,
        embeddingsize=256,
        attention='general',
        numlayers=2,
        rnn_class='lstm',
        learningrate=3,
        dropout=0.1,
        gradient_clip=0.1,
        lookuptable='enc_dec',
        optimizer='sgd',
        embedding_type='glove',
        momentum=0.9,
        bidirectional=False,
        context_length=-1,
        validation_every_n_secs=90,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=12,
        log_every_n_secs=10,
    )
    TrainLoop(parser).train()
