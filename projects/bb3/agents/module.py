#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BB3 Module Definition.
"""
from enum import Enum
from typing import Dict, List

import projects.bb3.constants as CONST


class Module(Enum):

    SEARCH_DECISION = 'sdm'
    MEMORY_DECISION = 'mdm'
    SEARCH_QUERY = 'sgm'
    MEMORY_GENERATOR = 'mgm'
    CONTEXTUAL_KNOWLEDGE = 'ckm'
    MEMORY_KNOWLEDGE = 'mkm'
    SEARCH_KNOWLEDGE = 'skm'
    CONTEXTUAL_DIALOGUE = 'crm'
    MEMORY_DIALOGUE = 'mrm'
    SEARCH_DIALOGUE = 'srm'
    VANILLA_DIALOGUE = 'vrm'
    GROUNDED_DIALOGUE = 'grm'
    OPENING_DIALOGUE = 'orm'

    @staticmethod
    def dialogue_modules() -> List['Module']:
        return [
            Module.CONTEXTUAL_DIALOGUE,
            Module.MEMORY_DIALOGUE,
            Module.SEARCH_DIALOGUE,
            Module.VANILLA_DIALOGUE,
            Module.GROUNDED_DIALOGUE,
            Module.OPENING_DIALOGUE,
        ]

    @staticmethod
    def knowledge_modules() -> List['Module']:
        return [
            Module.CONTEXTUAL_KNOWLEDGE,
            Module.MEMORY_KNOWLEDGE,
            Module.SEARCH_KNOWLEDGE,
        ]

    @staticmethod
    def decision_modules() -> List['Module']:
        return [
            Module.SEARCH_DECISION,
            Module.MEMORY_DECISION,
        ]

    def decision_do_key(self) -> str:
        return {
            Module.SEARCH_DECISION: 'search_decision_do_search_reply',
            Module.MEMORY_DECISION: 'memory_decision_do_access_reply',
        }[self]

    def decision_dont_key(self) -> str:
        return {
            Module.SEARCH_DECISION: 'search_decision_dont_search_reply',
            Module.MEMORY_DECISION: 'memory_decision_dont_access_reply',
        }[self]

    def message_name(self) -> str:
        """
        Name used to access output of a module in a Message.
        """
        return self.tag_to_agent()[self.value]

    def agent_name(self):
        """
        Display name for user's, and debugging, sake.
        """
        return f"{self.tag_to_agent()[self.value]}_agent"

    def model_file_path_key(self):
        """
        Opt key for model file path for this agent.
        """
        return f"{self.tag_to_agent()[self.value]}_response_model_path"

    def tag(self):
        return self.value

    def is_dialogue(self):
        return self in Module.dialogue_modules()

    def is_knowledge(self):
        return self in Module.knowledge_modules()

    def skip_search(self):
        return self.value not in ['mkm', 'skm']

    def is_one_turn_history(self):
        return self.value in ['mdm', 'sdm', 'mgm']

    @staticmethod
    def tag_to_agent() -> Dict[str, str]:
        return {
            'sdm': 'search_decision',
            'mdm': 'memory_decision',
            'sgm': 'search_query',
            'mgm': 'memory_generator',
            'ckm': 'contextual_knowledge',
            'mkm': 'memory_knowledge',
            'skm': 'search_knowledge',
            'crm': 'contextual_dialogue',
            'mrm': 'memory_dialogue',
            'srm': 'search_dialogue',
            'vrm': 'vanilla_dialogue',
            'grm': 'grounded_dialogue',
            'orm': 'opening_dialogue',
        }

    ##############
    # R2C2 Func. #
    ##############
    def r2c2_prompt(self):
        """
        Prompt token for this module.
        """
        return {
            'sdm': CONST.IS_SEARCH_REQUIRED,
            'mdm': CONST.IS_MEMORY_REQUIRED,
            'sgm': CONST.GENERATE_QUERY,
            'mgm': CONST.GENERATE_MEMORY,
            'mkm': CONST.ACCESS_MEMORY,
            'ckm': CONST.EXTRACT_ENTITY,
            'skm': CONST.GENERATE_KNOWLEDGE,
            'mrm': '',
            'crm': '',
            'srm': '',
            'vrm': '',
            'grm': '',
            'orm': '',
        }[self.value]

    def special_tokens(self):
        return {
            Module.CONTEXTUAL_KNOWLEDGE.message_name(): (
                CONST.BEGIN_ENTITY,
                CONST.END_ENTITY,
            ),
            Module.MEMORY_KNOWLEDGE.message_name(): (
                CONST.BEGIN_MEMORY,
                CONST.END_MEMORY,
            ),
            Module.SEARCH_KNOWLEDGE.message_name(): (
                CONST.TOKEN_KNOWLEDGE,
                CONST.TOKEN_END_KNOWLEDGE,
            ),
        }[self.tag_to_agent()[self.value]]

    #############
    # OPT Func. #
    #############
    def opt_prompt(self):
        """
        Prompt token for OPT models.
        """
        return {
            'sdm': "Person 2 must decide whether to search the internet.\n\n",
            'mdm': "A conversation between two persons. Person 2 must consult their notes about Person 1.\n\n",
            'sgm': "Person 2 must write a search query for a search engine.\n\n",
            'mgm': "A conversation between two persons. Person 2 writes a note about Person 1 to help remember information for later.\n\n",
            'ckm': "A conversation between two persons. Person 2 recalls a previous topic in the conversation.\n\n",
            'skm': "A conversation between two persons. Person 2 finds an interesting fact from the internet.\n\n",
            'mkm': "A conversation between two persons. Person 2 recalls an interesting fact about Person 1 or Person 2.\n\n",
            'crm': "A conversation between two persons. Person 2 would like to continue talking about a previous topic in the conversation.\n\n",
            'mrm': "A conversation between two persons. Person 2 would like to chat about an interesting fact about Person 1 or Person 2.\n\n",
            'srm': "A conversation between two persons. Person 2 would like to tell Person 1 about something Person 2 found on the internet.\n\n",
            'vrm': "A conversation between two persons.\n\n",
            'grm': "A conversation between two persons. Person 2 responds in a given style.\n\n",
            'orm': "A conversation between two persons. Person 2 begins the conversation given information about Person 1.\n\n",
        }[self.value]

    def opt_final_prefix(self):
        """
        Final prefix to put after constructing context for OPT.
        """
        import parlai_internal.projects.blenderbot3.agents.prompts as PROMPT

        return {
            'sdm': PROMPT.SEARCH_DECISION,
            'mdm': PROMPT.MEMORY_DECISION,
            'sgm': PROMPT.QUERY_GEN_PREFIX,
            'mgm': PROMPT.MEMORY_GEN_PREFIX,
            'mkm': PROMPT.MEMORY_KNOWLEDGE_PREFIX,
            'ckm': PROMPT.CONTEXTUAL_KNOWLEDGE_PREFIX,
            'skm': PROMPT.SEARCH_KNOWLEDGE_PREFIX,
            'mrm': PROMPT.SELF_PREFIX,
            'crm': PROMPT.SELF_PREFIX,
            'srm': PROMPT.SELF_PREFIX,
            'vrm': PROMPT.SELF_PREFIX,
            'grm': PROMPT.SELF_PREFIX,
            'orm': PROMPT.OPENING_PREFIX,
        }[self.value]

    def opt_shots(self) -> str:
        import projects.bb3.prompts as PROMPT

        return PROMPT.SHOTS[self]

    def opt_pre_context_tok(self):
        import parlai_internal.projects.blenderbot3.agents.prompts as PROMPT

        if self.is_knowledge() and self is not Module.CONTEXTUAL_KNOWLEDGE:
            return PROMPT.PRE_CONTEXT_TOK
        return ''

    def opt_post_context_tok(self):
        import parlai_internal.projects.blenderbot3.agents.prompts as PROMPT

        if self.is_dialogue() and self not in [
            Module.VANILLA_DIALOGUE,
            Module.OPENING_DIALOGUE,
        ]:
            return PROMPT.POST_CONTEXT_TOK
        return ''

    def opt_dialogue_knowledge_prefix(self):
        return {
            'mrm': "Personal Fact: ",
            'crm': "Previous Topic: ",
            'srm': "Interesting Fact: ",
        }[self.value]
