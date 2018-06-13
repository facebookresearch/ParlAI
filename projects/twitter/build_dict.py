# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Builds default dictionaries for twitter training and evaluation.
This provides a uniform setup for comparison between different models.

The eval_ppl script will use the dictionary returned by this build_dict()
function to conduct the perplexity evaluation, scoring only words that appear
in this dictionary.
"""


from parlai.scripts.build_dict import setup_args, build_dict as main_build_dict
DICT_FILE_30K = 'models:twitter/dict_30k'

def build_dict():
    # default is 30k
    return build_dict_30k()

def build_dict_30k():
    parser = setup_args()
    # first build on standard train and validation
    parser.set_defaults(
        task='twitter',
        dict_lower=True,
        dict_file=DICT_FILE_30K,
        dict_include_valid=True,
        dict_maxtokens=30000,
    )
    opt = parser.parse_args(args='')
    return main_build_dict(opt)


if __name__ == '__main__':
    build_dict()
