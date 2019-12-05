#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Example of training the sequence to sequence model on a bAbI task.
"""

from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == '__main__':
    parser = setup_args()
    # use set_defaults for setting args in py scripts so cmdline can overwrite
    # e.g. this file can be run with `-t babi:task10k:2` to use task 2 instead
    parser.set_defaults(
        task='babi:task10k:1',
        model='seq2seq',
        dict_file='/tmp/dict_babi_task1k_1',
        batchsize=32,
        validation_every_n_secs=30,
        validation_cutoff=0.95,  # "success" for babi tasks
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()
