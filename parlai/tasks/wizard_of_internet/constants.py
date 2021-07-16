#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET_NAME = 'wizard_of_interent'


# cli flag options
class HISTORY_TYPE:
    FULL = 'full'
    ONLY_LAST = 'onlylast'


# JSON file data dump keys
PERSONA = 'apprentice_persona'
DIALOG_HIST = 'dialog_history'
ACTION = 'action'
SPEAKER_ID = 'id'
MESSAGE_TEXT = 'text'
CONTEXT = 'context'
CONTENTS = 'contents'
SELECTED_CONTENTS = 'selected_contents'

ACTION_APPRENTICE_TO_WIZARD = 'Apprentice => Wizard'
ACTION_WIZARD_TO_APPRENTICE = 'Wizard => Apprentice'
ACTION_WIZARD_TO_SEARCH_AGENT = 'Wizard => SearchAgent'
ACTION_WIZARD_DOC_SELECTION = 'Wizard Doc Selection'
ACTION_SEARCH_AGENT_TO_WIZARD = 'SearchAgent => Wizard'
ACTION_ALL = 'All Actions'

# Message keys
TOTAL_CONVERSATION_INDEX = 'total_index'
SEARCH_QUERY = 'search_query'
RETRIEVED_DOCS = '__retrieved-docs__'
RETRIEVED_DOCS_URLS = '__retrieved-docs-urls__'
SELECTED_DOCS = '__selected-docs__'
SELECTED_DOCS_TITLES = '__select-docs-titles__'
SELECTED_SENTENCES = '__selected-sentences__'
SEARCH_RESULTS = 'search_results'
PARTNER_PREVIOUS_MESSAGE = 'partner_previous_msg'
IS_SEARCH_QUERY = 'is_search_query'
IS_LAST_SEARCH_QUERY = 'is_last_search_query'
LABELS = 'labels'

# Message values
NO_SEARCH_QUERY_USED = '__no_search_used__'
NO_RETRIEVED_DOCS_TOKEN = '__noretrieved-docs__'
NO_SELECTED_DOCS_TOKEN = '__noselected-docs__'
NO_SELECTED_SENTENCES_TOKEN = '__no_passages_used__'
NO_TITLE = '__no_title__'
NO_URLS = '__no_urls__'

# General values
WIZARD = 'wizard'
APPRENTICE = 'apprentice'
SEARCH_AGENT = 'search_agent'

# Flags/Opts default values
INCLUDE_PERSONA_DEFAULT = True
DIALOG_HIST_DEFAULT = HISTORY_TYPE.FULL
SKIP_ON_EMPTY_TEXT_DEFAULT = True
ONLY_LAST_QUERY_DEFAULT = False

# Tokens used in the teacher's action text filed
KNOWLEDGE_TOKEN = '__knowledge__'
END_KNOWLEDGE_TOKEN = '__endknowledge__'
AVAILABLE_KNOWLEDGE_TOKEN = '__available-knowledge__'
SELECTED_DOCS_TOKEN = '__selected-docs__'
SELECTED_SENTENCES_TOKEN = '__selected-sentences__'
