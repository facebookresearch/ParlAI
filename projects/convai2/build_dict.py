#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Builds the official dictionary for the ConvAI2 competition using the training and
validation sets for the 'convai2:self' task. The dictionary should contain 19304 tokens
after building.

All words in this dictionary should receive probability mass during perplexity
evaluation, or you will likely receive 'inf' perplexity.

Any tokens in the hidden test set which are not in this dictionary will not be
scored, so you do not have to try any schemes to assign probability to these
potential unknown words. See the evaluation script for more information.
"""


from parlai.scripts.build_dict import setup_args, build_dict as main_build_dict

DICT_FILE = 'models:convai2/dict_self'


def build_dict():
    parser = setup_args()
    # first build on standard train and validation
    parser.set_defaults(
        task='convai2:self',
        dict_lower=True,
        dict_file=DICT_FILE,
        dict_include_valid=True,
        dict_tokenizer='split',
    )
    opt = parser.parse_args(args="")
    return main_build_dict(opt)


if __name__ == '__main__':
    build_dict()
