#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Mutators for the SeeKeR Tasks.
"""
from abc import abstractmethod
import nltk
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, MessageMutator, ManyEpisodeMutator
from parlai.tasks.wizard_of_internet import constants as CONST
import parlai.tasks.wizard_of_internet.mutators  # type: ignore
import parlai.utils.logging as logging

try:
    import spacy
except ModuleNotFoundError:
    logging.error('Please install spacy: pip install spacy')
    spacy = None

from projects.seeker.utils import (
    GENERATE_QUERY,
    IS_SEARCH_REQUIRED,
    DO_SEARCH,
    DO_NOT_SEARCH,
    extract_entities,
    calc_f1_msmarco,
    calc_f1_msc,
)
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE

nlp = None

IGNORE_ENTITY_TYPES = set(
    ['CARDINAL', 'DATE', 'ORDINAL', 'PERCENT', 'QUANTITY', 'TIME']
)


@register_mutator('skip_retrieval_mutator')
class SkipRetrievalMutator(MessageMutator):
    """
    Mutator that adds a 'skip_retrieval' key to the observation.
    """

    def message_mutation(self, message: Message) -> Message:
        message.force_set('skip_retrieval', True)
        return message


@register_mutator('add_selected_sentences_mutator')
class AddSelectedSentencesMutator(MessageMutator):
    """
    Mutator that adds selected sentences to the messages.
    """

    def message_mutation(self, message: Message) -> Message:
        if 'checked_sentence' in message:
            message[CONST.SELECTED_SENTENCES] = [message['checked_sentence']]
        elif CONST.SELECTED_SENTENCES not in message:
            message[CONST.SELECTED_SENTENCES] = []
        return message


@register_mutator('add_retrieved_documents_mutator')
class AddRetrievedDocumentsMutator(MessageMutator):
    """
    Add retrieved docs and relevant keys.
    """

    def message_mutation(self, message: Message) -> Message:
        sentences = message['text'].split('\n')[:-1]
        if CONST.RETRIEVED_DOCS not in message:
            message[CONST.RETRIEVED_DOCS] = sentences
        if CONST.SELECTED_SENTENCES not in message:
            message[CONST.SELECTED_SENTENCES] = message['labels']
        if CONST.RETRIEVED_DOCS_URLS not in message:
            message[CONST.RETRIEVED_DOCS_URLS] = [''] * len(sentences)
        if CONST.RETRIEVED_DOCS_TITLES not in message:
            message[CONST.RETRIEVED_DOCS_TITLES] = [''] * len(sentences)
        return message


@register_mutator('prompt_search_query_mutator')
class PromptSearchQueryMutator(MessageMutator):
    """
    Add a __generate_search_query__ prompt to the end of the context, to inform the
    model.

    Assumes flattened data.
    """

    PROMPT = GENERATE_QUERY

    def message_mutation(self, message: Message) -> Message:
        if not message['text'].endswith(self.PROMPT):
            message.force_set('text', f"{message['text']} {self.PROMPT}")
        return message


@register_mutator('strip_context_mutator')
class StripContextMutator(MessageMutator):
    """
    Strip the context.

    This can be used to turn QA tasks to "open" QA tasks.
    """

    def message_mutation(self, message: Message) -> Message:
        message.force_set('text', message['text'].split('\n')[-1])
        return message


class SearchQueryClassificationMixin(MessageMutator):
    """
    Changes the message in the following ways:

    1. Makes the task *only one line of context*
    2. Adds a prompt to the end of the message, indicating a binary choice of __search-required__
    3. Changes the label to indicate whether to search or not.
    """

    PROMPT = IS_SEARCH_REQUIRED
    LABEL: str

    def message_mutation(self, message: Message) -> Message:
        assert self.get_label(message)
        if not message['text'].endswith(self.PROMPT):
            last_context = message['text'].split('\n')[-1]
            message.force_set('text', f"{last_context} {self.PROMPT}")
        if message['labels'] != [self.get_label(message)]:
            message.force_set('labels', [self.get_label(message)])
        message.pop('knowledge', None)
        return message

    @abstractmethod
    def get_label(self, message: Message) -> str:
        """
        Return the label.
        """


@register_mutator('do_generate_search_query_mutator')
class DoGenerateSearchQueryMutator(SearchQueryClassificationMixin):
    def get_label(self, message: Message) -> str:
        return DO_SEARCH


@register_mutator('dont_generate_search_query_mutator')
class DontGenerateSearchQueryMutator(SearchQueryClassificationMixin):
    def get_label(self, message: Message) -> str:
        return DO_NOT_SEARCH


@register_mutator('wow_maybe_generate_search_query_mutator')
class WowMaybeGenerateSearchQueryMutator(SearchQueryClassificationMixin):
    def get_label(self, message: Message) -> str:
        checked_sentence = message.get('checked_sentence')
        return DO_NOT_SEARCH if checked_sentence == 'no_passages_used' else DO_SEARCH


@register_mutator('woi_maybe_generate_search_query_mutator')
class WoiMaybeGenerateSearchQueryMutator(SearchQueryClassificationMixin):
    def get_label(self, message: Message) -> str:
        checked_sentences = ' '.join(message.get(CONST.SELECTED_SENTENCES))
        return (
            DO_NOT_SEARCH
            if checked_sentences == CONST.NO_SELECTED_SENTENCES_TOKEN
            else DO_SEARCH
        )


@register_mutator('bst_tasks_maybe_generate_search_query_mutator')
class BSTMaybeGenerateSearchQueryMutator(SearchQueryClassificationMixin):
    def get_label(self, message: Message) -> str:
        global nlp
        if nlp is None:
            assert spacy is not None
            nlp = spacy.load('en_core_web_sm')
        entities = nlp(message['text'].split('\n')[-1]).ents
        if entities:
            entities = [e for e in entities if e.label_ not in IGNORE_ENTITY_TYPES]
        return DO_NOT_SEARCH if not entities else DO_SEARCH


@register_mutator("woi_pop_documents_mutator")
class WoiRemoveDocuments(MessageMutator):
    """
    Remove documents from all message fields.

    Reduces space taken by the data.
    """

    def message_mutation(self, message: Message) -> Message:
        message.force_set(CONST.RETRIEVED_DOCS, [''])
        message.force_set(CONST.RETRIEVED_SENTENCES, [''])
        message.force_set(CONST.RETRIEVED_DOCS_TITLES, [''])
        message.force_set(CONST.RETRIEVED_DOCS_URLS, [''])
        message.force_set(CONST.SELECTED_DOCS, [''])
        message.force_set(CONST.SELECTED_DOCS_TITLES, [''])
        message.force_set(CONST.SELECTED_DOCS_URLS, [''])
        return message


@register_mutator("ms_marco_to_woi")
class MsMarcoToWoi(MessageMutator):
    """
    Turns context into docs amenable to FiD training.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        if not isinstance(new_message[CONST.RETRIEVED_DOCS], list):
            new_message.force_set(
                CONST.RETRIEVED_DOCS, [new_message[CONST.RETRIEVED_DOCS]]
            )
        new_message[CONST.RETRIEVED_DOCS_TITLES] = [''] * len(
            new_message[CONST.RETRIEVED_DOCS]
        )
        new_message[CONST.RETRIEVED_DOCS_URLS] = [''] * len(
            new_message[CONST.RETRIEVED_DOCS]
        )
        return new_message


@register_mutator("ms_marco_filter_has_answer")
class MsMarcoFilterHasAnswer(ManyEpisodeMutator):
    """
    Filters out examples that do not have an answer present.
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        for e in episode:
            if e.get('labels')[0] != 'No Answer Present.':
                out_episodes.append([e])
        return out_episodes


@register_mutator("ms_marco_create_fid_docs")
class MsMarcoCreateFidDocs(MessageMutator):
    """
    Turns context into docs amenable to FiD training.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        docs = message['text'].split('\n')
        context = docs[-1]
        docs = docs[0:-1]
        new_message[CONST.RETRIEVED_DOCS] = docs
        new_message.force_set('text', context)
        return new_message


@register_mutator("ms_marco_find_selected_sentence_for_knowledge")
class MsMarcoFindSelectedSentenceKnowledge(ManyEpisodeMutator):
    """
    Finds selected sentence for use in K2R.
    """

    target = 'knowledge'

    def many_episode_mutation(self, episode):
        out_episodes = []
        for m in episode:
            new_m = m.copy()
            gold_label = m.get('labels', [''])[0]
            try:
                gold_label_parts = nltk.word_tokenize(gold_label)
            except IndexError:
                # malformed gold label; continue
                continue
            best_f1 = 0
            best_sentence = ''
            docs = m.get(CONST.RETRIEVED_DOCS)
            for d in docs:
                try:
                    ds = nltk.sent_tokenize(d)
                except IndexError:
                    # malformed documents; continue as if it does not exist.
                    continue
                for s in ds:
                    f1 = calc_f1_msmarco(s, gold_label_parts)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_sentence = s
            ms_marco_f1_threshold = 0.5
            if self.target == 'knowledge':
                new_m.force_set('labels', [best_sentence])
            else:
                new_m.force_set(
                    'text',
                    new_m['text']
                    + f'\n{TOKEN_KNOWLEDGE} '
                    + best_sentence
                    + f' {TOKEN_END_KNOWLEDGE}',
                )

            new_m.force_set(CONST.SELECTED_SENTENCES, [best_sentence])
            if best_f1 > ms_marco_f1_threshold:
                out_episodes.append([new_m])
        return out_episodes


@register_mutator("ms_marco_find_selected_sentence_for_response")
class MsMarcoFindSelectedSentenceResponse(MsMarcoFindSelectedSentenceKnowledge):
    target = 'response'


@register_mutator("squad_to_woi")
class SquadToWoi(MessageMutator):
    """
    Turns context into docs amenable to FiD training.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        docs = message['text'].split('\n')
        context = docs[-1]
        docs = docs[0:-1]
        new_message[CONST.RETRIEVED_DOCS] = docs
        new_message[CONST.RETRIEVED_DOCS_TITLES] = [''] * len(docs)
        new_message[CONST.RETRIEVED_DOCS_URLS] = [''] * len(docs)
        new_message.force_set('text', context)
        return new_message


@register_mutator("triviaqa_to_woi")
class TriviaqaToWoi(MessageMutator):
    """
    Turns context into docs amenable to FiD training.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        docs = message['text'].split('\n')
        context = docs[-1]
        docs = [' '.join(docs[0:-1])]
        new_message[CONST.RETRIEVED_DOCS] = docs
        new_message[CONST.RETRIEVED_DOCS_TITLES] = ['']
        new_message[CONST.RETRIEVED_DOCS_URLS] = ['']
        new_message.force_set('text', context)
        return new_message


@register_mutator("nqopen_to_woi")
class NQOpenToWoi(MessageMutator):
    """
    Turns context into docs amenable to FiD training.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        new_message[CONST.RETRIEVED_DOCS] = [message['checked_sentence']]
        new_message[CONST.RETRIEVED_DOCS_TITLES] = [message['title']]
        new_message[CONST.RETRIEVED_DOCS_URLS] = ['']
        new_message.pop('checked_sentence')
        if message.get('history', '') != '':
            text = message.get('history', '') + '\n' + message.get('text', '')
            new_message.force_set('text', text)
        return new_message


@register_mutator("nq_to_woi")
class NQToWoi(ManyEpisodeMutator):
    """
    Turns context into docs amenable to FiD training.

    Note we currently throw away all examples with no label.
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        for message in episode:
            if message['labels'][0] == '':
                pass
            else:
                new_message = message.copy()
                docs = message['text'].split('\n')
                context = docs[-1]
                docs = docs[0:-1]
                new_message[CONST.RETRIEVED_DOCS] = docs
                new_message[CONST.RETRIEVED_DOCS_TITLES] = ['']
                new_message[CONST.RETRIEVED_DOCS_URLS] = ['']
                new_message.force_set('text', context)
                out_episodes.append([new_message])
        return out_episodes


@register_mutator("extract_entity_for_knowledge_model")
class ExtractEntity(ManyEpisodeMutator):
    """
    Picks an entity in the context and uses it as the intended target.
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        for message in episode:
            new_message = message.copy()
            context_except_last_line = '\n'.join(message['text'].split('\n')[:-1])
            context_ent = extract_entities(context_except_last_line)
            label_ent = extract_entities(message['labels'][0])
            ents = set(context_ent).intersection(label_ent)
            if len(ents) > 0:
                longest_ent = max(ents, key=len)
            else:
                continue
            new_message.force_set('response_labels', message['labels'])
            new_message.force_set('labels', [longest_ent])
            new_message.force_set(CONST.SELECTED_SENTENCES, [longest_ent])
            new_message.force_set(
                CONST.RETRIEVED_DOCS, message['text'].split('\n')[:-1]
            )
            blanks = [''] * len(message['text'].split('\n')[:-1])
            new_message.force_set(CONST.RETRIEVED_DOCS_URLS, blanks)
            new_message.force_set(CONST.RETRIEVED_DOCS_TITLES, blanks)
            out_episodes.append([new_message])
        return out_episodes


@register_mutator("extract_entity_for_response_model")
class ExtractEntityResponse(ManyEpisodeMutator):
    """
    Picks an entity in the context and uses it as the intended target.
    """

    BEGIN_KNOWLEDGE: str = CONST.KNOWLEDGE_TOKEN
    END_KNOWLEDGE: str = CONST.END_KNOWLEDGE_TOKEN

    def many_episode_mutation(self, episode):
        out_episodes = []
        for message in episode:
            new_message = message.copy()
            context_except_last_line = '\n'.join(message['text'].split('\n')[:-1])
            context_ent = extract_entities(context_except_last_line)
            label_ent = extract_entities(message['labels'][0])
            ents = set(context_ent).intersection(label_ent)
            if len(ents) > 0:
                longest_ent = max(ents, key=len)
            else:
                continue
            knol_text = f'{self.BEGIN_KNOWLEDGE} {longest_ent} {self.END_KNOWLEDGE}'
            new_text = message['text'] + '\n' + knol_text
            new_message.force_set('text', new_text)
            out_episodes.append([new_message])
        return out_episodes


@register_mutator("msc_find_selected_sentence_knowledge")
class MscFindSelectedSentenceKnowledge(ManyEpisodeMutator):
    """
    Finds selected sentence for use in K2R.
    """

    target = 'knowledge'

    def many_episode_mutation(self, episode):
        out_episodes = []
        for m in episode:
            new_m = m.copy()
            gold_label = m.get('labels', [''])[0]
            try:
                gold_label_parts = nltk.word_tokenize(gold_label)
            except IndexError:
                # malformed label
                gold_label_parts = []
            best_f1 = 0
            best_sentence = ''
            context_except_last_line = '\n'.join(m['text'].split('\n')[:-1])
            docs = [context_except_last_line]
            for d in docs:
                ds = d.split('\n')
                for s in ds:
                    f1 = calc_f1_msc(s, gold_label_parts)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_sentence = s
            if best_sentence != '':
                # find which speaker it is
                z = m['text'].split('\n')
                z.reverse()
                ind = z.index(best_sentence)
                if (ind % 2) == 0:
                    speaker = '__them__'
                else:
                    speaker = '__you__'
                if 'your persona:' in best_sentence:
                    speaker = '__you__'
                best_sentence = best_sentence + " " + speaker
            msc_f1_threshold = 0.3
            if self.target == 'knowledge':
                new_m.force_set('labels', [best_sentence])
            else:
                new_m.force_set(
                    'text',
                    new_m['text']
                    + '\n__knowledge__ '
                    + best_sentence
                    + ' __endknowledge__',
                )
            new_m.force_set('old_target', gold_label)

            if best_f1 > msc_f1_threshold:
                out_episodes.append([new_m])
        return out_episodes


@register_mutator("msc_find_selected_sentence_response")
class MscFindSelectedSentenceResponse(MscFindSelectedSentenceKnowledge):
    """
    Finds selected sentence for use in K2R.
    """

    target = 'response'
