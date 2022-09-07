#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from projects.seeker.utils import (
    GENERATE_QUERY,
    IS_SEARCH_REQUIRED,
    DO_SEARCH,
    DO_NOT_SEARCH,
)

from parlai.tasks.msc.agents import NOPERSONA, DUMMY_TEXT
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE


### Special Tokens
GENERATE_KNOWLEDGE = '__generate-knowledge__'
GENERATE_MEMORY = '__generate-memory__'
ACCESS_MEMORY = '__access-memory__'
EXTRACT_ENTITY = '__extract-entity__'
IS_MEMORY_REQUIRED = '__is-memory-required__'
DO_ACCESS_MEMORY = '__do-access-memory__'
DONT_ACCESS_MEMORY = '__do-not-access-memory__'
BEGIN_MEMORY = '__memory__'
END_MEMORY = '__endmemory__'
BEGIN_ENTITY = '__entity__'
END_ENTITY = '__endentity__'
YOU = '__you__'
THEM = '__them__'
BEGIN_STYLE = '__style__'
END_STYLE = '__end-style__'
YOU_PREFIX = 'You: '
PARTNER_PREFIX = 'Partner: '


ALL_SPECIAL_TOKENS = [
    GENERATE_QUERY,
    IS_SEARCH_REQUIRED,
    DO_SEARCH,
    DO_NOT_SEARCH,
    TOKEN_KNOWLEDGE,
    TOKEN_END_KNOWLEDGE,
    NOPERSONA,
    DUMMY_TEXT,
    GENERATE_KNOWLEDGE,
    GENERATE_MEMORY,
    ACCESS_MEMORY,
    EXTRACT_ENTITY,
    IS_MEMORY_REQUIRED,
    DO_ACCESS_MEMORY,
    DONT_ACCESS_MEMORY,
    BEGIN_MEMORY,
    END_MEMORY,
    BEGIN_ENTITY,
    END_ENTITY,
    YOU,
    THEM,
    YOU_PREFIX,
    PARTNER_PREFIX,
]
