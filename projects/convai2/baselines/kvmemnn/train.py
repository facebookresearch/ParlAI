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
        task='convai2:self_revised',
        evaltask='convai2:self',
        model='projects.personachat.kvmemnn.kvmemnn:KvmemnnAgent',
        model_file='/tmp/persona_self_rephrase',
        dict_lower=True,
        dict_include_valid=True,
        dict_maxexs=-1,
        datatype='train',
        hops=1,
        lins=0,
        embeddingsize=1000,
        learningrate=0.1,
        share_embeddings=True,
        margin=0.1,
        tfidf=False,
        max_train_time=28800,
        validation_every_n_secs=900,
        validation_metric='accuracy',
        validation_metric_mode='max',
        validation_patience=-1,
        validation_max_examples=100000,
        log_every_n_secs=10,
        numthreads=40,
        dict_tokenizer='split',
    )
    TrainLoop(parser.parse_args()).train()
