# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train DrQA model on SQuAD.
"""

from parlai.scripts.train_model import setup_args, TrainLoop

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='squad:index',
        evaltask='squad',
        model='drqa',
        model_file='/tmp/model_drqa',
        embedding_file='models:glove_vectors/glove.840B.300d.txt',
        max_train_time=28800,
        validation_every_n_secs=1000,
        validation_metric='accuracy',
        validation_metric_mode='max',
        validation_patience=-1,
        validation_max_examples=100000,
        log_every_n_secs=10,
        batchsize=32,
    )
    TrainLoop(parser.parse_args()).train()
