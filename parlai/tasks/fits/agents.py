#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import json
import random
from typing import Optional

from parlai.utils.misc import warn_once
from parlai.utils.data import DatatypeHelper
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from parlai.core.message import Message
from parlai.utils.io import PathManager


from parlai.tasks.wizard_of_internet import constants as woi_consts
from .mutators import (
    normalize_reply,
    get_knowledge_appended_context,
    get_knowledge_sentence_with_special_tokens_only,
)
from .build import build
from .constants import (
    FeedbackType,
    TITLE_PASSAGE_DELIMITER,
    OK_LABEL,
    NOT_OK_LABEL,
)


def format_to_woi_consistent(docs, woi_format_gold_docs):
    retrieved_docs = []
    retrieved_doc_titles = []
    for doc_str in docs:
        doc_splits = doc_str.split(TITLE_PASSAGE_DELIMITER)
        doc_passage = TITLE_PASSAGE_DELIMITER.join(doc_splits[1:])
        retrieved_docs.append(doc_passage)
        retrieved_doc_titles.append(doc_splits[0])

    woi_format_gold_docs.update(
        {
            woi_consts.RETRIEVED_DOCS: retrieved_docs,
            woi_consts.RETRIEVED_DOCS_TITLES: retrieved_doc_titles,
            woi_consts.RETRIEVED_DOCS_URLS: ['' for _ in retrieved_docs],
        }
    )
    return woi_format_gold_docs


def dedup_list(arr):
    new_arr = []
    for e in arr:
        if e not in new_arr:
            new_arr.append(e)
    return new_arr


class FitsBaseTeacher(DialogTeacher):
    """
    Base Teacher that extract metadata for dialogues.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Fits Base options')
        agent.add_argument(
            '--metrics-per-feedback',
            type=str,
            default=None,
            help='metrics to show when breaking down by last feedback that takes effect; if None then do not break down by feedbacktype',
        )
        agent.add_argument(
            '--add-bot-failure-to-feedback-input',
            type=bool,
            default=False,
            help='If True, then include bot failure turns in the context, otherwise don"t ',
        )
        agent.add_argument(
            '--add-knowledge-sentence-to-feedback-input',
            type=bool,
            default=False,
            help='If True, then include knowledge sentence in the context, otherwise don"t ',
        )
        agent.add_argument(
            '--fits-task-version',
            type=str,
            default='v1',
            help='dataset version, v1 = 7.5k dialogues, v2 = 13.5k dialogues',
        )
        agent.add_argument(
            '--unseen-task',
            type=bool,
            default=False,
            help='dataset skills seen by default and unseen if set True',
        )
        return parser

    def __init__(self, opt, shared=None):
        opt['datafile'] = build(opt)
        self.is_training = DatatypeHelper.is_training(opt['datatype'])
        self.metrics_per_feedback = opt.get('metrics_per_feedback', None)
        self.add_bot_failure_to_feedback_input = opt.get(
            'add_bot_failure_to_feedback_input', False
        )
        self.add_knowledge_sentence_to_feedback_input = opt.get(
            'add_knowledge_sentence_to_feedback_input', False
        )
        self.unseen_task = opt.get('unseen_task', False)
        super().__init__(opt, shared)
        self.id = f'fits_{opt["mutators"]}_{opt["fits_task_version"]}_seen{opt["unseen_task"]}'

    def _load_raw_data(self, data_path):
        print('loading: ' + data_path)

        if self.datatype.startswith('train'):
            path_to_open = os.path.join(data_path, 'train.txt')
        elif self.datatype.startswith('valid'):
            path_to_open = os.path.join(data_path, 'valid.txt')
        else:
            path_to_open = os.path.join(data_path, 'test.txt')

        if self.unseen_task:
            assert self.datatype.startswith('test')
            path_to_open = os.path.join(data_path, 'test.txt')
            warn_once('No train/valid splits for unseen tasks')

        with PathManager.open(path_to_open) as f:
            raw_data = [json.loads(line.strip()) for line in f]

        return raw_data

    def setup_data(self, data_path):
        raw_data = self._load_raw_data(data_path)
        data = []
        for chat in raw_data:
            perfect_episode, _ = self._get_perfect_episode(
                chat, is_training=self.is_training
            )
            data.append(perfect_episode)

        for episode in data:
            start_idx = 0
            for i, turn in enumerate(episode):
                yield Message(turn), i == start_idx

    def _add_feedback_to_perfect_utt(
        self, bot_acts, human_acts, perfect_dialog_context
    ):
        feedback_chain = [
            bot_act.get('improve_bucket', FeedbackType.PERFECT.value)
            for bot_act in bot_acts
        ]
        feedback_examples = []
        effective_human_feedback = {}
        human_revision = {'search_query': [], 'selected_sentences': []}
        bot_gold = {'search_query': []}
        bot_no_feedback = {'search_query': []}
        bot_modular_failure = {'search_query': []}
        woi_format_gold_docs = {
            woi_consts.SELECTED_SENTENCES: []
        }  # otherwise get_retrieved_knowledge will break
        if 'search_queries' in bot_acts[0]:
            woi_format_gold_docs = format_to_woi_consistent(
                bot_acts[0]['top_docs'], woi_format_gold_docs
            )

        if len(bot_acts) == 1:
            # no feedback turn
            if 'search_queries' in bot_acts[0]:
                bot_no_feedback['search_query'].append(bot_acts[0]['search_queries'])
        else:
            for i, bot_act in enumerate(bot_acts):
                # skip updating the top_docs or search_queries if the bot response is from human better_response
                if (
                    i > 0
                    and bot_acts[i - 1].get('improve_bucket', '')
                    != FeedbackType.CORRECT_RESPONSE.value
                    and 'top_docs' in bot_act
                    and isinstance(bot_act['top_docs'], list)
                ):
                    woi_format_gold_docs = format_to_woi_consistent(
                        bot_act['top_docs'], woi_format_gold_docs
                    )
                # clear the knowledge sentence selection if it's from a fresh search
                if i > 0 and bot_acts[i - 1].get('improve_bucket', '') in [
                    FeedbackType.CORRECT_SEARCH_QUERY.value
                ]:
                    woi_format_gold_docs[woi_consts.SELECTED_SENTENCES] = []

                if (
                    bot_act.get('improve_bucket', '')
                    == FeedbackType.CORRECT_SEARCH_QUERY.value
                ):
                    effective_human_feedback = {
                        'search_query': bot_acts[i + 1]['search_queries']
                    }
                    human_revision['search_query'].append(
                        effective_human_feedback['search_query']
                    )
                    if 'search_queries' in bot_act:
                        bot_modular_failure['search_query'].append(
                            bot_act['search_queries']
                        )
                elif (
                    bot_act.get('improve_bucket', '') == FeedbackType.CORRECT_DOC.value
                ):
                    if 'better_response' in effective_human_feedback:
                        del effective_human_feedback['better_response']
                    effective_human_feedback['selected_sentences'] = human_acts[i + 1][
                        'feedback_text'
                    ]
                    human_revision['selected_sentences'].append(
                        effective_human_feedback['selected_sentences']
                    )
                    woi_format_gold_docs[woi_consts.SELECTED_SENTENCES] = [
                        effective_human_feedback['selected_sentences']
                    ]
                    bot_gold['search_query'].append(bot_acts[i]['search_queries'])
                elif (
                    bot_act.get('improve_bucket', '')
                    == FeedbackType.CORRECT_RESPONSE.value
                ):
                    effective_human_feedback['better_response'] = human_acts[i + 1][
                        'feedback_text'
                    ]

                # collect feedback examples as target
                if bot_act.get('improve_bucket', '') in [
                    FeedbackType.CORRECT_SEARCH_QUERY.value,
                    FeedbackType.CORRECT_DOC.value,
                    FeedbackType.CORRECT_RESPONSE.value,
                ]:
                    assert (
                        human_acts[i + 1]['feedback_type'] == bot_act['improve_bucket']
                    )
                    new_context_texts = [normalize_reply(human_acts[0]['text'])]
                    if self.add_bot_failure_to_feedback_input:
                        if (
                            self.add_knowledge_sentence_to_feedback_input
                            and woi_consts.SELECTED_SENTENCES in woi_format_gold_docs
                            and woi_format_gold_docs[woi_consts.SELECTED_SENTENCES]
                        ):
                            new_context_texts.append(
                                get_knowledge_sentence_with_special_tokens_only(
                                    woi_format_gold_docs[woi_consts.SELECTED_SENTENCES][
                                        -1
                                    ],
                                    normalize_knowledge=True,
                                )
                            )
                        new_context_texts.append(normalize_reply(bot_act['text']))

                    feedback_examples.append(
                        {
                            'text': '\n'.join(
                                perfect_dialog_context + [t for t in new_context_texts]
                            ),
                            'labels': [normalize_reply(human_acts[i + 1]['text'])],
                            'last_nonperfect_feedback': bot_act['improve_bucket'],
                            'bot_failure_text': bot_act['text'],
                            **(copy.deepcopy(woi_format_gold_docs)),
                        }
                    )
                    if 'search_queries' in bot_act:
                        feedback_examples[-1]['search_queries'] = bot_act[
                            'search_queries'
                        ]
                    if 'selected_sentences' in effective_human_feedback:
                        feedback_examples[-1][
                            'selected_sentences'
                        ] = effective_human_feedback['selected_sentences']

        effective_human_feedback['feedback_chain'] = feedback_chain
        effective_human_feedback['last_nonperfect_feedback'] = (
            feedback_chain[-2] if len(bot_acts) > 1 else FeedbackType.NONE.value
        )

        # remove redundant examples
        human_revision['search_query'] = dedup_list(human_revision['search_query'])
        bot_gold['search_query'] = dedup_list(
            [
                q
                for q in bot_gold['search_query']
                if q not in human_revision['search_query']
            ]
        )
        human_revision['selected_sentences'] = dedup_list(
            human_revision['selected_sentences']
        )

        perfect_bot_utt_with_feedback = {
            'text': bot_acts[-1]['text'],
            'human_revision': human_revision,
            'bot_gold': bot_gold,
            'bot_no_feedback': bot_no_feedback,
            'bot_modular_failure': bot_modular_failure,
            'human_acts': human_acts,
            'bot_acts': bot_acts,
        }
        for k, v in effective_human_feedback.items():
            if v is not None:
                perfect_bot_utt_with_feedback[k] = v

        if woi_format_gold_docs:
            for k, v in woi_format_gold_docs.items():
                if k in perfect_bot_utt_with_feedback:
                    raise RuntimeError("name collision in continual learning task")
                perfect_bot_utt_with_feedback[k] = v

        return [
            {'text': human_acts[0]['text']},
            perfect_bot_utt_with_feedback,
        ], feedback_examples

    def _get_perfect_episode(self, chat, is_training):
        raw_dialog = chat['dialog']
        assert len(raw_dialog) % 2 == 0, "odd number of utterances"
        perfect_dialog = []
        # store all bot/human acts before the bot generate the next perfect response
        bot_acts = []
        human_acts = []
        perfect_dialog_context = []
        raw_feedback_examples = []
        for i, utt in enumerate(raw_dialog):
            # whether it's human message or bot message
            if utt.get('id') in ['YOU', 'Speaker 1']:
                human_acts.append(utt)
            else:
                if (
                    i == len(raw_dialog) - 1
                    and utt.get('improve_bucket', FeedbackType.PERFECT.value)
                    != FeedbackType.PERFECT.value
                ):
                    utt['improve_bucket'] = FeedbackType.PERFECT.value
                bot_acts.append(utt)
                if (
                    utt.get('improve_bucket', FeedbackType.PERFECT.value)
                    == FeedbackType.PERFECT.value
                    or i == len(raw_dialog) - 1
                ):
                    (
                        utts_with_feedbacks,
                        feedback_examples,
                    ) = self._add_feedback_to_perfect_utt(
                        bot_acts, human_acts, perfect_dialog_context
                    )
                    perfect_dialog.extend(utts_with_feedbacks)
                    perfect_dialog_context.extend(
                        [normalize_reply(t['text']) for t in utts_with_feedbacks]
                    )
                    raw_feedback_examples.extend(feedback_examples)
                    bot_acts = []
                    human_acts = []

        perfect_episode = []
        chat_metadata = {
            'specific_task': chat['chat_task_name'],
            'task_requirement': chat['chat_task_requirement'],
            'topic': chat['topic'],
            'generic_topic': chat['generic_topic'],
            'domain': chat['domain'],
            'task_uid': chat['chat_task_uid'],
        }
        for i in range(0, len(perfect_dialog) - 1, 2):
            action = {
                'id': self.id,
                'text': normalize_reply(perfect_dialog[i]['text']),
                'labels': [normalize_reply(perfect_dialog[i + 1]['text'])],
            }
            for k, v in perfect_dialog[i + 1].items():
                if k not in ['text', 'id', 'labels']:
                    action[k] = v
            action.update(chat_metadata)
            perfect_episode.append(action)

        feedback_examples_with_metadata = []
        for act in raw_feedback_examples:
            feedback_examples_with_metadata.append(
                {
                    **act,
                    **chat_metadata,
                }
            )

        return perfect_episode, feedback_examples_with_metadata

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        obs = super().observe(observation)
        if self.last_act is None:
            return obs
        has_update = False
        if (
            self.metrics_per_feedback is not None
            and self.last_act.get('last_nonperfect_feedback', None) is not None
        ):
            for key, value in observation['metrics'].items():
                if (
                    self.metrics_per_feedback == 'all'
                    or key in self.metrics_per_feedback
                ):
                    self.metrics.add(
                        f"{self.last_act['last_nonperfect_feedback']}_{key}", value
                    )
                    has_update = True

        if has_update:
            updated_recent_metrics = self.metrics.report_recent()
            if updated_recent_metrics and 'metrics' in obs:
                obs.pop('metrics')
                obs['metrics'] = updated_recent_metrics
        return obs


class FitsFeedbackTeacher(FitsBaseTeacher):
    """
    Teacher that display free-form text feedbacks.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = f'fits_feedback_{opt["mutators"]}_botTurn{self.add_bot_failure_to_feedback_input}'

    def setup_data(self, data_path):
        raw_data = self._load_raw_data(data_path)

        data = []
        for chat in raw_data:
            _, feedback_examples_with_metadata = self._get_perfect_episode(
                chat, is_training=self.is_training
            )
            data.extend(feedback_examples_with_metadata)

        for turn in data:
            message = Message(turn)
            yield message, True


class FitsSatisfactionTeacher(FitsBaseTeacher):
    """
    Teacher that display binary classes __ok__, __notok__ on bot response.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group(
            'Continual Learning New Tasks Satisfaction Teacher options'
        )
        agent.add_argument(
            '--prepend-task-specifics',
            type=bool,
            default=False,
            help='whether to prepend task name and task requirement to the dialogue context',
        )
        agent.add_argument(
            '--must-have-knowledge-sentence',
            type=bool,
            default=False,
            help='whether to only include examples with perfect/gold knowledge sentence',
        )
        agent.add_argument(
            '--positive-example-source',
            type=str,
            default='all',
            choices=['human_gold', 'human_and_bot_gold_revision', 'all'],
            help='source of positive examples concatenated by "," ',
        )
        return parser

    def __init__(self, opt, shared=None):
        opt_cp = copy.deepcopy(opt)
        opt_cp['add_bot_failure_to_feedback_input'] = True
        self.must_have_knowledge_sentence = opt_cp['must_have_knowledge_sentence']
        if self.must_have_knowledge_sentence:
            opt_cp['add_knowledge_sentence_to_feedback_input'] = True
        if (
            not self.must_have_knowledge_sentence
            and opt_cp['add_knowledge_sentence_to_feedback_input']
        ):
            warn_once(
                "You are loading Satisfaction Teacher with a mixture of examples both w/ and w/o knowledge"
            )
        self.prepend_task_specifics = opt_cp['prepend_task_specifics']
        self.positive_example_source = opt_cp.get('positive_example_source', 'all')
        super().__init__(opt_cp, shared)
        self.id = f'fits_satisfaction_{opt_cp["mutators"]}_prependtask_{self.prepend_task_specifics}_{self.positive_example_source}'

    def setup_data(self, data_path):
        raw_data = self._load_raw_data(data_path)

        data = []
        neg_data = []
        pos_data = []

        def _get_prepend_task_info(ex):
            return (
                '__task__ '
                + normalize_reply(ex['specific_task'])
                + ' | '
                + normalize_reply(ex['task_requirement'])
                + '__endtask__'
            )

        for chat in raw_data:
            (
                perfect_episode,
                feedback_examples_with_metadata,
            ) = self._get_perfect_episode(chat, is_training=self.is_training)
            previous_texts = []
            for ex in feedback_examples_with_metadata:
                if self.must_have_knowledge_sentence:
                    if not (
                        woi_consts.SELECTED_SENTENCES in ex
                        and ex[woi_consts.SELECTED_SENTENCES]
                    ):
                        continue
                text = ex['text']
                if self.prepend_task_specifics:
                    task_context = _get_prepend_task_info(ex)
                    if task_context:
                        text = task_context + '\n' + text
                # dedup sample
                if text in previous_texts:
                    continue
                previous_texts.append(text)
                new_exs = copy.deepcopy(ex)
                new_exs.update(
                    {
                        'text': text,
                        'labels': [NOT_OK_LABEL],
                        'last_nonperfect_feedback': ex['last_nonperfect_feedback'],
                    }
                )
                neg_data.append(new_exs)
            history = []
            for turn in perfect_episode:
                history.append(turn['text'])
                history.append(turn['labels'][0])
                text = '\n'.join(history)
                if self.must_have_knowledge_sentence:
                    if not (
                        woi_consts.SELECTED_SENTENCES in turn
                        and turn[woi_consts.SELECTED_SENTENCES]
                    ):
                        continue
                    else:
                        text = get_knowledge_appended_context(
                            '\n'.join(history[:-1]),
                            turn[woi_consts.SELECTED_SENTENCES][-1],
                            normalize_knowledge=True,
                        )
                        text = '\n'.join([text] + history[-1:])
                if self.prepend_task_specifics:
                    task_context = _get_prepend_task_info(turn)
                    if task_context:
                        text = task_context + '\n' + text
                if (
                    self.positive_example_source == 'human_gold'
                    and turn['last_nonperfect_feedback']
                    != FeedbackType.CORRECT_RESPONSE.value
                ) or (
                    self.positive_example_source == 'human_and_bot_gold_revision'
                    and turn['last_nonperfect_feedback'] == FeedbackType.NONE.value
                ):
                    continue
                new_exs = copy.deepcopy(turn)
                new_exs.update(
                    {
                        'text': text,
                        'labels': [OK_LABEL],
                        'last_nonperfect_feedback': FeedbackType.NONE.value,
                    }
                )
                pos_data.append(new_exs)

        data = neg_data + pos_data
        random.shuffle(data)
        for turn in data:
            message = Message(turn)
            yield message, True


class QueryTeacher(FitsBaseTeacher):
    """
    Query for finetune query generator.

    human_gold: human written search query that lead to perfect utterance,
        e.g better_search_query->perfect; better_search_query -> better_search_doc -> perfect;
            better_search_query -> better_search_doc -> better_response -> perfect
    human_bronze: human written search query that doesn't belong to perfect utterance
    bot_gold: bot search queries that leads to perfect utterance e.g. better_doc -> perfect
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group(
            'Continual Learning New Tasks Query Teacher options'
        )
        agent.add_argument(
            '--query-source',
            type=str,
            default='human_gold',
            choices=[
                'human_gold',
                'human_bronze',
                'bot_gold',
                'human',
                'all',
            ],
            help='metrics to show when breaking down by bot nickname',
        )
        return parser

    def __init__(self, opt, shared=None):
        self.query_source = opt.get('query_source', None)
        super().__init__(opt, shared)
        self.id = f'fits_query_{opt["mutators"]}'

    def setup_data(self, data_path):
        raw_data = self._load_raw_data(data_path)
        data = []
        for chat in raw_data:
            perfect_episode, _ = self._get_perfect_episode(
                chat, is_training=self.is_training
            )
            feedbacks = self._extract_feedback(perfect_episode)
            if self.query_source in ['human_gold', 'human', 'all']:
                data.extend(feedbacks['human_gold'])
            if self.query_source in ['human_bronze', 'human', 'all']:
                data.extend(feedbacks['human_bronze'])
            if self.query_source in ['bot_gold', 'all']:
                data.extend(feedbacks['bot_gold'])

        for turn in data:
            yield Message(turn), True

    def _extract_feedback(self, episode):
        history = []
        res = {
            'human_gold': [],
            'human_bronze': [],
            'bot_gold': [],
            'bot_failure': [],
        }

        for turn in episode:
            history.append(normalize_reply(turn['text']))
            human_provided_queries = turn['human_revision']['search_query']
            per_turn_res = {
                'human_gold': [],
                'human_bronze': [],
                'bot_gold': [],
                'bot_failure': [],
            }
            if (
                turn['last_nonperfect_feedback']
                == FeedbackType.CORRECT_SEARCH_QUERY.value
                or (
                    # ..., better_search_query, better_search_doc, perfect
                    len(turn['feedback_chain']) >= 3
                    and turn['feedback_chain'][-3]
                    == FeedbackType.CORRECT_SEARCH_QUERY.value
                    and turn['feedback_chain'][-2] == FeedbackType.CORRECT_DOC.value
                )
                or (
                    # ..., better_search_query, better_search_doc, better_others, perfect
                    len(turn['feedback_chain']) >= 4
                    and turn['feedback_chain'][-4]
                    == FeedbackType.CORRECT_SEARCH_QUERY.value
                    and turn['feedback_chain'][-3] == FeedbackType.CORRECT_DOC.value
                    and turn['feedback_chain'][-2]
                    == FeedbackType.CORRECT_RESPONSE.value
                )
            ):
                per_turn_res['human_gold'] = [human_provided_queries[-1]]
            per_turn_res['human_bronze'] = [
                q for q in human_provided_queries if q not in per_turn_res['human_gold']
            ]
            per_turn_res['bot_gold'] = turn['bot_gold'].get('search_query', [])
            per_turn_res['bot_failure'] = turn['bot_modular_failure'].get(
                'search_query', []
            )

            for source in per_turn_res:
                for q in per_turn_res[source]:
                    res[source].append(
                        {'text': '\n'.join(history), 'labels': [normalize_reply(q)]}
                    )
            history.append(normalize_reply(turn['labels'][0]))
        return res


class DefaultTeacher(FitsBaseTeacher):
    pass
