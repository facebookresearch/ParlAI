#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Builds default dictionaries for twitter training and evaluation. This provides a uniform
setup for comparison between different models.

The eval_ppl script will use the dictionary returned by this build_dict() function to
conduct the perplexity evaluation, scoring only words that appear in this dictionary.
"""

DICT_FILE_30K = 'models:twitter/dict/dict_30k'
