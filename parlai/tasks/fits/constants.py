#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class FeedbackType(Enum):
    PERFECT = "perfect"
    CORRECT_SEARCH_QUERY = "better_search_query"
    CORRECT_DOC = "better_search_doc"
    CORRECT_HISTORY = "better_history"
    CORRECT_RESPONSE = "better_others"
    NONE = 'none'


NO_SEARCH = '__NO__SEARCH__'
TITLE_PASSAGE_DELIMITER = ' | '

OK_LABEL = '__ok__'
NOT_OK_LABEL = '__notok__'

UNWANTED_TOKENS = ['_potentially_unsafe__', '_pots and pots_']
