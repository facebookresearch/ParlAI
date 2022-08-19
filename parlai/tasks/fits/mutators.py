#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from string import punctuation
import copy
from typing import List

from parlai.core.mutators import register_mutator, ManyEpisodeMutator, MessageMutator
from parlai.core.message import Message
from parlai.mutators.flatten import FlattenMutator
from parlai.utils.strings import normalize_reply as parlai_normalize_reply

from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from parlai.tasks.wizard_of_internet import constants as consts

from .constants import (
    FeedbackType,
    UNWANTED_TOKENS,
)


def normalize_reply(x):
    normalized_x = parlai_normalize_reply(x).strip()
    for t in UNWANTED_TOKENS:
        if normalized_x.lower().endswith(t):
            normalized_x = normalized_x[: -len(t)].strip()
    if '\n' in normalized_x:
        multilines = [
            ln.strip() for ln in normalized_x.split('\n') if len(ln.strip()) > 0
        ]
        if len(multilines) > 0:
            multilines = [
                (ln + ',' if ln[-1] not in punctuation else ln) for ln in multilines
            ]
            normalized_x = ' '.join(multilines)
            if normalized_x[-1] == ',':
                normalized_x = normalized_x[:-1] + '.'
    return normalized_x


def get_knowledge_sentence_with_special_tokens_only(
    knowledge_sentence: str, normalize_knowledge=False
):
    text = knowledge_sentence
    if normalize_knowledge:
        text = normalize_reply(text)
    return f"{TOKEN_KNOWLEDGE} {text} {TOKEN_END_KNOWLEDGE}"


def get_knowledge_appended_context(
    context: str, knowledge_sentence: str, normalize_knowledge=False
):
    return (
        context
        + "\n"
        + get_knowledge_sentence_with_special_tokens_only(
            knowledge_sentence, normalize_knowledge=normalize_knowledge
        )
    )


class FeedbackMixin(FlattenMutator):
    """
    Filter examples by last feedback, by subclassing FlattenMutator it is equivalent to.

    --mutators flatten+<this_mutator>
    """

    target_feedback: str

    def many_episode_mutation(self, episode):
        episodes = super().many_episode_mutation(episode)
        for episode in episodes:
            if episode[-1]['last_nonperfect_feedback'] == self.target_feedback:
                yield episode


@register_mutator("flatten_gold_human_mutator")
class GoldHuman(FeedbackMixin):
    """
    Filter examples by last nonperfect feedback.
    """

    target_feedback = FeedbackType.CORRECT_RESPONSE.value


@register_mutator("flatten_gold_bot_by_better_search_query_mutator")
class GoldBotByBetterSearchQuery(FeedbackMixin):
    """
    Filter examples by last nonperfect feedback.
    """

    target_feedback = FeedbackType.CORRECT_SEARCH_QUERY.value


@register_mutator("flatten_gold_bot_by_better_search_doc_mutator")
class GoldBotByBetterSearchDoc(FeedbackMixin):
    """
    Filter examples by last nonperfect feedback.
    """

    target_feedback = FeedbackType.CORRECT_DOC.value


@register_mutator("flatten_gold_bot_no_feedback_mutator")
class GoldBotNoFeedback(FeedbackMixin):
    """
    Filter examples by last nonperfect feedback.
    """

    target_feedback = FeedbackType.NONE.value


@register_mutator("flatten_gold_knowledge_response_mutator")
class GoldKnowledge(FlattenMutator):
    """
    give gold knowledge.
    """

    TOKEN_ENC_KNOWLEDGE = '__encknowledge__ '
    TOKEN_DEC_KNOWLEDGE = '__decknowledge__ '

    def many_episode_mutation(self, episode):
        episodes = super().many_episode_mutation(episode)
        for episode in episodes:
            assert len(episode) == 1
            if 'selected_sentences' in episode[-1]:
                new_episode = copy.deepcopy(episode)
                new_episode[-1].force_set(
                    'labels',
                    [
                        self.TOKEN_DEC_KNOWLEDGE
                        + ' '
                        + normalize_reply(episode[-1]['selected_sentences'])
                    ],
                )
                new_episode[-1].force_set(
                    'text', new_episode[-1]['text'] + ' ' + self.TOKEN_ENC_KNOWLEDGE
                )
                yield new_episode


@register_mutator("flatten_gold_knowledge_response_no_special_token_mutator")
class GoldKnowledgeNoSpecialToken(FlattenMutator):
    """
    give gold knowledge as target w/o special tokens.
    """

    def many_episode_mutation(self, episode):
        episodes = super().many_episode_mutation(episode)
        for episode in episodes:
            assert len(episode) == 1
            if 'selected_sentences' in episode[-1]:
                new_episode = copy.deepcopy(episode)
                new_episode[-1].force_set(
                    'labels',
                    [normalize_reply(episode[-1]['selected_sentences'])],
                )
                yield new_episode


@register_mutator("knowledge_appended_mutator")
class ContLnKnowledgeAppendedDialogueResponseTeacher(ManyEpisodeMutator):
    """
    parlai dd -t fits:mutators=flatten+knowledge_appended_mutator_internal appended gold
    knowleddge to response.
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        assert (
            len(episode) == 1
        ), 'knowledge_appended_mutator must be applied on flattened episode'
        for message in episode:
            if 'selected_sentences' in message:
                new_message = message.copy()
                new_message.force_set(
                    'text',
                    get_knowledge_appended_context(
                        new_message['text'],
                        new_message['selected_sentences'],
                        normalize_knowledge=True,
                    ),
                )
                out_episodes.append([new_message])
        return out_episodes


@register_mutator('no_woi_gold_docs_mutator')
class NoWoIGoldDocTeacher(MessageMutator):
    """
    Mutator that throw away WoI gold doc fields so the OBSERVATION_ECHO_RETRIEVER
    doesn't return top_docs.
    """

    def message_mutation(self, message: Message) -> Message:
        for key in [
            consts.RETRIEVED_DOCS_URLS,
            consts.RETRIEVED_DOCS_TITLES,
            consts.RETRIEVED_DOCS,
            'top_docs',
        ]:
            if key in message:
                del message[key]
        return message


@register_mutator('feedback_special_token_mutator')
class FeedbackSpecialToken(MessageMutator):
    """
    Mutator that throw away WoI gold doc fields so the OBSERVATION_ECHO_RETRIEVER
    doesn't return top_docs.
    """

    TOKEN_ENC_FEEDBACK = '__encfeedback__ '
    TOKEN_DEC_FEEDBACK = '__decfeedback__ '

    def _get_special_token_with_feedbacktypes(self, feedback_type):
        return self.TOKEN_DEC_FEEDBACK

    def message_mutation(self, message: Message) -> Message:
        message.force_set('text', message['text'] + ' ' + self.TOKEN_ENC_FEEDBACK)
        message.force_set(
            'labels',
            [
                self._get_special_token_with_feedbacktypes(
                    message['last_nonperfect_feedback']
                )
                + ' '
                + message['labels'][0]
            ],
        )
        return message


@register_mutator('feedback_type_specific_special_token_mutator')
class FeedbackTypeSpecificSpecialToken(FeedbackSpecialToken):
    """
    Mutators that add.
    """

    def _get_special_token_with_feedbacktypes(self, feedback_type):
        feedback_type = ''.join(feedback_type.split('_'))
        return f'__decfeedback{feedback_type}__'


@register_mutator('bot_failure_appended_mutator')
class BotFailureSpecialToken(MessageMutator):
    """
    Mutator that add the bot failure text in text (with special tokens surrounded)
    """

    TOKEN_BOT_FAILURE = '__botfailure__ '
    TOKEN_BOT_FAILURE_END = '__endbotfailure__ '

    def message_mutation(self, message: Message) -> Message:
        if 'bot_failure_text' in message:
            new_message = message.copy()
            new_message.force_set(
                'text',
                message['text']
                + f"\n{self.TOKEN_BOT_FAILURE} {normalize_reply(new_message['bot_failure_text'])} {self.TOKEN_BOT_FAILURE_END}",
            )
            return new_message

        return message


class SingleClassMixin(ManyEpisodeMutator):
    """
    Filter examples by class label.

    --mutators <this_mutator>
    """

    target_class: str

    def many_episode_mutation(self, episode):
        out_episodes = []
        for message in episode:
            if self.target_class in message['labels']:
                out_episodes.append([message])
        return out_episodes


@register_mutator("ok_mutator")
class OkClassFilter(SingleClassMixin):
    target_class = '__ok__'


@register_mutator("notok_mutator")
class NotOkClassFilter(SingleClassMixin):
    target_class = '__notok__'


@register_mutator('ok_pos_notok_neg_mutator')
class PosNegLabelConvertor(MessageMutator):
    """
    Mutator that throw away WoI gold doc fields so the OBSERVATION_ECHO_RETRIEVER
    doesn't return top_docs.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = copy.deepcopy(message)
        if message['labels'][0] == '__ok__':
            new_message.force_set('labels', ['pos'])
        elif message['labels'][0] == '__notok__':
            new_message.force_set('labels', ['neg'])
        return new_message


@register_mutator('fits_director_LTR_mutator')
class DirectorLeftToRightMutator(ManyEpisodeMutator):
    """
    DirectorLeftToRightMutator prepares data for training left to right (LTR) classifier
    for Director model.

    This limits to context to all but last utterance that is fed to the encoder.
    The final utterance is considered as a label for the decoder and the attribute/classifier
    labels are stored seperately marking the final utterance pos. or neg.

    This mutator also adds a is_ltr flag to differentiate classifier exs from the generator exs which are used to finetune the generator model.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        new_episodes = []
        for message in episode:
            text = message['text']
            utterances = text.split('\n')

            if len(utterances) < 2:
                continue

            new_message = copy.deepcopy(message)
            new_message.force_set('is_ltr', True)
            if message['labels'][0] == 'pos':
                new_message.force_set('classifier_label', 'pos')
            else:
                new_message.force_set('classifier_label', 'neg')
            new_message.force_set('text', '\n'.join(utterances[:-1]))
            new_message.force_set('labels', [utterances[-1]])
            new_episodes.append([new_message])
        return new_episodes


@register_mutator('director_skip_reranking_mutator')
class DirectorSkipModififyingLogprobMutator(MessageMutator):
    """
    Mutator that adds a 'skip_director_reranking' key.
    """

    def message_mutation(self, message: Message) -> Message:
        message.force_set('skip_director_reranking', True)
        return message
