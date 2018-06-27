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

DICT_FILE_30K = 'models:twitter/dict/dict_30k'
