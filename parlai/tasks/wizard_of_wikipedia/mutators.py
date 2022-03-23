#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.mutators import register_mutator, MessageMutator, ManyEpisodeMutator
from parlai.core.message import Message
import parlai.tasks.wizard_of_internet.constants as CONST
import random


@register_mutator("wow_add_checked_sentence_to_input")
class AddCheckedSentence(MessageMutator):
    """
    Adds the checked sentence to the end of the text.

    But only a single time.
    """

    @property
    def checked_sentence_kword(self):
        return 'checked_sentence'

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        if 'text' not in message:
            return message
        text = new_message.pop('text')
        checked_sentence = new_message.get(self.checked_sentence_kword, '')
        if isinstance(checked_sentence, list):
            checked_sentence = ' '.join(checked_sentence)

        text += (
            f'\n{CONST.KNOWLEDGE_TOKEN} {checked_sentence} {CONST.END_KNOWLEDGE_TOKEN}'
        )
        new_message['text'] = text

        return new_message


@register_mutator("wow_checked_sentence_as_label")
class CheckedSentenceAsLabel(MessageMutator):
    """
    Uses the checked sentence (knowledge) as label.
    """

    @property
    def checked_sentence_kword(self):
        return 'checked_sentence'

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        if 'text' not in message or 'labels' not in message or not message['labels']:
            return message
        labels = new_message.pop('labels')
        checked_sentence = new_message.get(self.checked_sentence_kword, '')
        if isinstance(checked_sentence, list):
            checked_sentence = ' '.join(checked_sentence)

        new_message['dialogue_response'] = labels
        new_message['labels'] = [checked_sentence]
        return new_message


@register_mutator("wow_add_label_to_input")
class AddLabel(MessageMutator):
    """
    Adds the dialogue sentence to the input.

    But only a single time.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        if 'text' not in message or 'labels' not in message or not message['labels']:
            return message
        if 'dialogue_response' in new_message:
            # checked_sentence_as_label was applied before
            labels = new_message['dialogue_response']
        else:
            labels = new_message['labels']
        dialogue_response = labels[0]
        text = new_message.pop('text')

        text += f'\n{CONST.TOKEN_LABEL} {dialogue_response} {CONST.TOKEN_END_LABEL}'
        new_message['text'] = text

        return new_message


@register_mutator("wow_add_label_to_input_lm")
class AddLabelLM(MessageMutator):
    """
    Adds the dialogue sentence to the input (language modeling version).

    Language modeling version where a random piece of the label is sampled in
    the input. The rest is placed inside special tokens.

    E.g. run with: parlai display_data -t wizard_of_wikipedia -n 100 -dt valid --mutators
    flatten,add_label_to_input_lm_wow

    To add the checked sentence as the label, use:
        parlai display_data -t wizard_of_wikipedia -n 100 -dt valid --mutators
        flatten,add_label_to_input_lm_wow,checked_sentence_as_label_wow
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        if 'text' not in message or 'labels' not in message or not message['labels']:
            return message
        if 'dialogue_response' in new_message:
            # checked_sentence_as_label was applied before
            labels = new_message['dialogue_response']
        else:
            labels = new_message['labels']
        dialogue_response = labels[0]
        text = new_message.pop('text')

        ls = dialogue_response.split()
        ind = random.randint(0, len(ls) - 1)
        label1 = ' '.join(ls[0:ind])
        label2 = ' '.join(ls[ind : len(ls)])

        text += f'\n{label1}\n{CONST.TOKEN_LABEL} {label2} {CONST.TOKEN_END_LABEL}'
        new_message['text'] = text

        return new_message


@register_mutator("wow_filter_no_passage_used")
class WowFilterNoPassageUsed(ManyEpisodeMutator):
    """
    Allows to filter any examples where no passage was selected to base the wizard reply
    on.
    """

    def fails(self, sentences: str) -> bool:
        """
        Return if checked sentence is filtered.
        """
        return sentences == "no_passages_used"

    def many_episode_mutation(self, episode):
        out_episodes = []
        for e in episode:
            checked_sentences = e.get('checked_sentence')
            if self.fails(checked_sentences):
                pass
            else:
                out_episodes.append([e])
        return out_episodes


@register_mutator("wow_only_no_passage_used")
class WowOnlyNoPassageUsed(WowFilterNoPassageUsed):
    """
    Allows to filter such that only examples where no passage was selected are used.
    """

    def fails(self, sentences: str) -> bool:
        """
        Return if checked sentence is filtered.
        """
        return sentences != "no_passages_used"


@register_mutator("wow_to_woi")
class WowToWoi(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        new_message.pop('knowledge')
        new_docs = [' '.join(message['knowledge'].split('\n'))]
        new_titles = ['']
        new_urls = ['']
        new_message.force_set(CONST.RETRIEVED_DOCS, new_docs)
        new_message.force_set(CONST.RETRIEVED_DOCS_TITLES, new_titles)
        new_message.force_set(CONST.RETRIEVED_DOCS_URLS, new_urls)
        return new_message
