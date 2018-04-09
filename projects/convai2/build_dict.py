# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Builds the official dictionary for the ConvAI2 competition using the
training and validation sets for the 'convai2:self' task.
The dictionary should contain 19304 tokens after building.

All words in this dictionary should receive probability mass during perplexity
evaluation, or you will likely receive 'inf' perplexity.

Any tokens in the hidden test set which are not in this dictionary will not be
scored, so you do not have to try any schemes to assign probability to these
potential unknown words. See the evaluation script for more information.
"""


from examples.build_dict import setup_args, build_dict as main_build_dict

def build_dict():
    DICT_FINAL = 'models:convai2/dict_self'

    parser = setup_args()
    # first build on standard train and validation
    parser.set_defaults(
        task='convai2:self',
        dict_lower=True,
        dict_file=DICT_FINAL,
        dict_include_valid=True,
    )
    opt = parser.parse_args()
    return main_build_dict(opt)


if __name__ == '__main__':
    build_dict()
