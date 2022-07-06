#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from typing import Iterable, List
from nltk.stem import PorterStemmer
from parlai.crowdsourcing.utils.acceptability import (
    AcceptabilityChecker,
    normalize_answer,
)
import parlai.utils.logging as logging


# Bad persona violations
PERSONA_REPEATS_PROMPT = 'repeated the prompt text'
ASKED_WIZARD_QUESTION = 'asked wizard in the persona details'
COPIED_EXTENDED_PERSONA = 'extended persona copies the main persona'
GENERIC_EXTENDED_PERSONA = 'extended persona is generic'

QUESTION_PHRASE = 'what is your'

# Wizard knowledge violations
DEFAULT_KNOWLEDGE_OVERLAP_THRESHOLD = 0.05

POOR_SEARCH_QUERIES = 'poor search queries'
IRRELEVANT_SEARCH__QUERIES = 'irrelevant search terms'
NOT_ENOUGH_SEARCH = 'not enough selected knowledge sources'
SELECTED_SHORT_PIECES = 'short knowledge pieces selected.'
LOW_KNOWLEDGE_OVERLAP = 'low knowledge overlap'


def tokenize_text(text, stemmer, as_set=True):
    text = normalize_answer(text)
    tokens = [stemmer.stem(word) for word in text.split(' ')]
    if as_set:
        tokens = set(tokens)
    return tokens


def overlap_ratios(a: set, b: set) -> float:
    """
    Calculates the Jacard distance between two sets.
    """
    overlap = a.intersection(b)
    union = a.union(b)
    return len(overlap) / (len(union) + 0.001)


def is_valid_agent_chat_message(message, agent_id):
    return (
        message.get('text')
        and message.get('id') == agent_id
        and not message.get('is_search_query', False)
    )


def bad_persona(persona, stemmer):
    """
    Check for poor persona selection by apprentice.
    """
    persona_parts = persona.split('\n')

    # It is not from the persona selection ones (personas used during the pilot).
    if not (
        len(persona_parts) == 2
        or (len(persona_parts) == 3 and 'I live in ' in persona_parts[0])
    ):
        logging.warning(f'Old fashioned persona: {persona}')
        return

    # Removing the location ('I live in X') part
    if len(persona_parts) == 3:
        persona_parts = persona_parts[1:]

    main_pers, ext_pers = [p.lower() for p in persona_parts]

    violations = []

    # Bad main persona response
    if main_pers.startswith('My favorite '):
        for phrase in ('i like', 'my favorite'):
            persona_core = main_pers
            # Remove the original My favorite
            persona_core = main_pers[len('My favorite ') :]
            if phrase in persona_core.lower():
                violations.append(PERSONA_REPEATS_PROMPT)
                break

    # Extended persona that asks questions
    for phrase in (QUESTION_PHRASE,):
        if phrase in ext_pers:
            violations.append(ASKED_WIZARD_QUESTION)

    # Extended persona that mostly repeats the main persona
    main_pers_tokens = tokenize_text(main_pers, stemmer)
    ext_pers_tokens = tokenize_text(ext_pers, stemmer)
    if len(ext_pers_tokens.difference(main_pers_tokens)) < 2:
        violations.append(COPIED_EXTENDED_PERSONA)

    # Use of non-generic words in persona.
    common_phrases = ('i', 'it', 'like', 'very', 'much', 'favorite', 'is', 'am')
    tokens = [w.strip() for w in ext_pers.split(' ') if w]
    ext_useful_words = [t for t in tokens if t not in common_phrases]
    if len(tokens) > 4 and len(ext_useful_words) < 2:
        violations.append(GENERIC_EXTENDED_PERSONA)

    return violations


def poor_knowledge_selection(messages, persona, stemmer, knwldg_ovlp_thrshld):
    """
    Check for poor search and knowledge selection by wizard.
    """
    # Collecting search and knowledge selections
    search_terms = []
    selected_knowledge = []
    message_history_tokens = tokenize_text(persona, stemmer)

    n_search_query_not_in_history = 0
    for msg in messages:
        if msg.get('text', None):
            message_history_tokens = message_history_tokens.union(
                tokenize_text(msg['text'], stemmer)
            )

        if msg['id'] != 'Wizard':
            continue

        selections = msg.get('task_data', {}).get('selected_text_candidates')
        if not selections or selections[0][0]:
            continue

        search_query = msg['task_data']['search_query']
        search_terms.append(search_query)
        if message_history_tokens.isdisjoint(tokenize_text(search_query, stemmer)):
            n_search_query_not_in_history += 1

        selected_parts = []
        for doc_id in range(1, len(selections)):
            doc_selections = selections[doc_id]
            for sentence_id in range(len(doc_selections)):
                if doc_selections[sentence_id]:
                    selected_parts.append(
                        msg['task_data']['text_candidates'][doc_id - 1]['content'][
                            sentence_id
                        ]
                    )

        selected_knowledge.append(
            {'text': msg['text'], 'knowledge': ' '.join(selected_parts)}
        )

    knowledge_length = []
    knowledge_overlaps = []
    for knwldg in selected_knowledge:
        knowledge_tokens = tokenize_text(knwldg['knowledge'], stemmer)
        knowledge_length.append(len(knowledge_tokens))

        response_tokens = tokenize_text(knwldg['text'], stemmer)
        knowledge_overlaps.append(overlap_ratios(knowledge_tokens, response_tokens))

    violations = []

    # Repeated the same search queries
    if len(search_terms) - len(set(search_terms)) > 3:
        violations.append(POOR_SEARCH_QUERIES)

    # Search doesn't have overlap with message history
    if n_search_query_not_in_history > 2:
        violations.append(IRRELEVANT_SEARCH__QUERIES)

    # No selection
    if not knowledge_length:
        violations.append(NOT_ENOUGH_SEARCH)

    # Only selecting short sentences
    if np.average(knowledge_length) < 5:
        violations.append(SELECTED_SHORT_PIECES)

    # Small overlap between response and the selected knowledge parts
    knowledge_overlap_avg = np.average(knowledge_overlaps)
    if knowledge_overlap_avg < knwldg_ovlp_thrshld:
        violations.append(f'{LOW_KNOWLEDGE_OVERLAP} ({knowledge_overlap_avg})')

    return violations


class WizardOfInternetAcceptabilityChecker(AcceptabilityChecker):
    """
    ParlAI general acceptabilty checker customized for the wizard of internet.
    """

    def __init__(self):
        self.knowledge_overlap_threshold = DEFAULT_KNOWLEDGE_OVERLAP_THRESHOLD
        self.post_stemmer = PorterStemmer()
        super().__init__()

    def check_messages(
        self,
        agent_id: str,
        persona: str,
        messages: List[str],
        is_worker_0: bool,
        violation_types: Iterable[str] = (),
    ) -> str:
        violations = []
        general_chat_violations = super().check_messages(
            self.get_conversation_messages(messages, agent_id),
            is_worker_0,
            violation_types,
        )
        if general_chat_violations:
            violations.extend(general_chat_violations.split(','))

        if agent_id == 'Apprentice':
            persona_violations = bad_persona(persona, self.post_stemmer)
            if persona_violations:
                violations.extend(persona_violations)

        if agent_id == 'Wizard':
            knowledge_violations = poor_knowledge_selection(
                messages, persona, self.post_stemmer, self.knowledge_overlap_threshold
            )
            if knowledge_violations:
                violations.extend(knowledge_violations)

        return ','.join(violations)

    def get_conversation_messages(self, agent_messages, agent_id):
        return [
            msg['text']
            for msg in agent_messages
            if is_valid_agent_chat_message(msg, agent_id)
        ]
