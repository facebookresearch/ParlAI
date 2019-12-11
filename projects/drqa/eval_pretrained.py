#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Evaluate a pre-trained reader model on the SQuAD:FullDocTeacher.
"""
from parlai.scripts.eval_model import setup_args, eval_model


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        task='squad:fulldoc', model='drqa', model_file='models:drqa/squad/model'
    )
    opt = parser.parse_args(print_args=False)
    eval_model(opt, print_parser=parser)
