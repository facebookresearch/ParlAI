# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate a pre-trained reader model on the SQuAD:FullDocTeacher.
"""
from parlai.scripts.eval_model import setup_args, eval_model


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        task='squad:fulldoc',
        model='drqa',
        model_file='models:drqa/squad/model'
    )
    opt = parser.parse_args(print_args=False)
    eval_model(opt, print_parser=parser)
