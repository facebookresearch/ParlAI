#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e # stop if any tests fail
cd ../examples/
python display_data.py -t babi:task1k:1
python base_train.py -t babi:task1k:1
python display_data.py -t babi:task1k:1,squad -n 100
python eval_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid -n 10
python display_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid -n 10
python build_dict.py -t babi:task1k:1 --dict-file /tmp/dict.tsv
python train_model.py -m seq2seq -t babi:task1k:1 -bs 8 -e 1 -mf /tmp/model_s2s

# if this returns without an error code, you're good!
python train_model.py -m drqa -t squad -bs 32 -mf /tmp/model_drqa & sleep 60 ; kill $!
