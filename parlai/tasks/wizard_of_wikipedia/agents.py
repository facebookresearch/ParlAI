#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A dataset with conversations directly grounded with knowledge retrieved from Wikipedia.
Contains 201k utterances from 22k dialogues spanning over 1300 diverse topics, split
into train, test, and valid sets. The test and valid sets are split into two sets each:
one with overlapping topics with the train set, and one with unseen topics.

To access the different valid/test splits (unseen/seen), specify
the corresponding split (`random_split` for seen, `topic_split`
for unseen) after the last colon in the task.
E.g. `wizard_of_wikipedia:WizardDialogKnowledgeTeacher:random_split`
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, normalize_answer, F1Metric
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from parlai.utils.io import PathManager
from parlai.utils import logging
from parlai.utils.misc import warn_once
from .build import build

import json
import os
import random


TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'


def _first_val(dictionary):
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ''


def _first_key(dictionary):
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ''


def _get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.

    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = _first_val(title_dict) if title_dict else ''
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[1:-1])
            if (
                cand_title1
                and cand_title1 in k_dict
                and sentence in k_dict[cand_title1]
            ):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


def _path(opt, split='random_split'):
    build(opt)
    dp = os.path.join(opt['datapath'], 'wizard_of_wikipedia')
    dt = opt.get('datatype', 'train').split(':')[0]
    if dt == 'train':
        df = 'train.json'
    else:
        df = '{}_{}.json'.format(dt, split)
    return os.path.join(dp, df)


class RareWordF1Calculator:
    """
    Helper class for computing F1 with an emphasis on infrequent words.
    """

    def __init__(self, corpus: str, top_p: float = 0.5):
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        words = normalize_answer(corpus).split()
        self._freq_dist = nltk.FreqDist(words)
        self._cutoff_count = RareWordF1Calculator._find_cutoff_count(
            self._freq_dist, top_p
        )

    @property
    def freq_dist(self):
        return self._freq_dist

    @staticmethod
    def _find_cutoff_count(freq_dist, top_p: float) -> int:
        """
        Finds the word occurance for which the cumulative occurances are `top_p` of the
        overall word count.
        """
        assert top_p < 1
        target = sum(freq_dist.values()) * top_p
        cumul = 0
        for _, v in freq_dist.most_common():
            cumul += v
            if cumul > target:
                return v
        raise RuntimeError(f"Invalid top {top_p*100}% of the corpus distribution")

    @staticmethod
    def _filter(freq_dist, cutoff: int, text: str) -> str:
        """
        For words that are found in the reference distribution, filters those with an
        occurrence count less than the cutoff.
        """
        words = normalize_answer(text).split()
        return " ".join([w for w in words if freq_dist.get(w, cutoff) < cutoff])

    def compute(self, guess: str, answers: Iterable[str]) -> F1Metric:
        if guess is None or answers is None:
            return F1Metric(0, 0)
        guess = RareWordF1Calculator._filter(self._freq_dist, self._cutoff_count, guess)
        answers = [
            RareWordF1Calculator._filter(self._freq_dist, self._cutoff_count, a)
            for a in answers
        ]
        if not any(len(a) for a in answers):
            # no rare words in labels, set denominator to zero
            return F1Metric(0, 0)
        return F1Metric.compute(guess, answers)


def _build_rare_word_f1(datapath: str) -> RareWordF1Calculator:
    all_text = ''
    data_path = os.path.join(datapath, 'wizard_of_wikipedia', 'data.json')
    with PathManager.open(data_path) as f:
        data = json.load(f)
        all_text += ' '.join(m['text'] for d in data for m in d['dialog']) + ' '
    return RareWordF1Calculator(all_text, top_p=0.5)


class WizardOfWikipediaTeacher(FixedDialogTeacher):
    """
    The default teacher; essentially reads the json file and outputs the raw data.

    Actions have the following form:
    {
        'wizard_eval': <evaluation of wizard>,
        'chosen_topic': <chosen_topic>,
        'chosen_topic_passage': <chosen topic passage>,
        'mtdo': <whether the conversation had sufficient overlap>,
        'text': <text>
        'retrieved_topics': <topics retrieved for text>
        'full_retrieved_passages': <full retrieved passages>
        'retrieved_passages': <passages shown to turker>
        'checked_sentence': <checked sentence if wizard, else None>
        'checked_passage': <checked_passage if wizard, else None>
    }

    The 'passages' are lists of 1 entry dicts, mapping a topic to the sentences

    Specify the valid/test split after the last colon in the task, e.g.
    wizard_of_wikipedia:<teacher>:random_split
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        task = opt.get('task', 'wizard_of_wikipedia:WizardOfWikipedia:random_split')
        split = task.split(':')
        split = split[2] if len(split) == 3 else 'random_split'
        opt['task'] = 'wizard_of_wikipedia'
        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.data_path = _path(opt, split=split)
            self._setup_data()
        self.num_exs = sum(len(d['dialog']) for d in self.data)
        self.reset()

    def _setup_data(self):
        print('loading: ' + self.data_path)
        with PathManager.open(self.data_path) as f:
            self.data = json.load(f)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        dialog_entry = d['dialog'][entry_idx]
        episode_done = entry_idx == len(d['dialog']) - 1
        action = {
            'wizard_eval': d['wizard_eval'],
            'chosen_topic': d['chosen_topic'],
            'chosen_topic_passage': d['chosen_topic_passage'],
            'text': dialog_entry['text'],
            'retrieved_topics': dialog_entry['retrieved_topics'],
            'retrieved_passages': dialog_entry['retrieved_passages'],
            'checked_sentence': dialog_entry.get('checked_sentence', None),
            'checked_passage': dialog_entry.get('checked_passage', None),
            'episode_done': episode_done,
        }

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


###############################################################
#                                                             #
# Dialog Teachers                                             #
#                                                             #
###############################################################


class WizardDialogKnowledgeTeacher(WizardOfWikipediaTeacher):
    """
    Teacher that returns the following action dict:
    {
        'text': chosen_topic\n # if first ex in ep
                last_apprentice_message\n # if possible
                wizard_message # if --label-type is chosen_sent

        'knowledge': title_1 sentence_1\n
                            .
                            .
                            .
                     title_m sentence_n # all knowledge available to wizard
        'labels': [title_checked sentence_checked] # default
                                    OR
                  [wizard_response] # if --label-type set to 'response'

        'label_candidates': knowledge + [no_passages_used no_passages_used]
    }
    """

    def __init__(self, opt, shared=None):
        self.add_missing_turns = opt.get('add_missing_turns', 'none')
        super().__init__(opt, shared)
        self.label_type = opt.get('label_type', 'response')
        self.include_knowledge = opt.get('include_knowledge', True)
        self.include_checked_sentence = opt.get('include_checked_sentence', False)
        self.knowledge_separator = opt.get('include_knowledge_separator', False)
        self.chosen_topic_delimiter = opt.get('chosen_topic_delimiter', '\n')
        self.num_exs = sum(self.len_episode(i) for i in range(len(self.data)))
        if shared and 'rare_word_f1' in shared:
            self.rare_word_f1 = shared['rare_word_f1']
        elif self.label_type == 'response':
            self.rare_word_f1 = _build_rare_word_f1(opt['datapath'])

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Wizard Dialog Knowledge arguments')
        agent.add_argument(
            '--label-type',
            type=str,
            choices=['response', 'chosen_sent'],
            default='response',
            help='whether to populate label field with the '
            'wizard response, or the chosen sentence',
        )
        agent.add_argument(
            '--include-knowledge',
            type='bool',
            default=True,
            help='Whether to include the knowledge available to' ' the wizard',
        )
        agent.add_argument(
            '--include-checked-sentence',
            type='bool',
            default=True,
            help='Whether to include the Wizard\'s' 'checked sentence',
        )
        agent.add_argument(
            '--include-knowledge-separator',
            type='bool',
            default=False,
            help='include special __knowledge__ token between ' 'title and passage',
        )
        agent.add_argument(
            '--chosen-topic-delimiter',
            type=str,
            default='\n',
            help='delimiter used when including chosen topic',
        )
        agent.add_argument(
            '--num-topics',
            type=int,
            default=5,
            help='in interactive mode, this is the number of topic choices'
            'the human will have',
        )
        agent.add_argument(
            '--add-missing-turns',
            type=str,
            choices=['none', 'train', 'all'],
            default='none',
            help='For reproducibility, the default "none" is the previous version which misssing some data.'
            'When "train" is chosen, only the training set is supplemented.'
            'When "all" is chosen, all data are supplemented.',
        )
        return parser

    def share(self):
        shared = super().share()
        if hasattr(self, 'rare_word_f1'):
            shared['rare_word_f1'] = self.rare_word_f1
        return shared

    def len_episode(self, ep):
        d = self.data[ep]
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        if wizard_first:
            if self.add_missing_turns == 'none':
                warn_once(
                    'Some data not being used. If you are not trying to reproduce '
                    'the previous results, it is recommended that you run with the '
                    'flag --add-missing-turns train or --add-missing-turns all.'
                )
                len_ep = (len(d['dialog']) - 1) // 2
            elif self.add_missing_turns == 'train' and 'train' not in self.datatype:
                len_ep = (len(d['dialog']) - 1) // 2
            else:
                len_ep = (len(d['dialog']) - 1) // 2 + 1
            return len_ep
        return len(d['dialog']) // 2

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

        # first, get knowledge
        apprentice_ret_passages = wizard_ret_passages = {}

        if not wizard_first or idx != 0:
            apprentice_entry = d['dialog'][idx - 1]
            apprentice_ret_passages = apprentice_entry['retrieved_passages']
        if idx - 2 >= 0:
            wizard_prev_entry = d['dialog'][idx - 2]
            wizard_ret_passages = wizard_prev_entry['retrieved_passages']

        chosen_topic = d.get('chosen_topic', '')
        chosen_topic_passages = d['chosen_topic_passage']
        chosen_topic = d.get('chosen_topic', '')

        knowledge_dict = {chosen_topic: chosen_topic_passages}
        for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
            for passage in ret_passes:
                for k, v in passage.items():
                    if k not in knowledge_dict.keys():
                        knowledge_dict[k] = v

        # then, get text
        if idx == 0:
            # first message - only have the chosen topic
            text = chosen_topic
        elif idx == 1:
            # first response - only have the first message
            text = (
                f"{chosen_topic}{self.chosen_topic_delimiter}{apprentice_entry['text']}"
            )
        else:
            text = ''
            if self.label_type == 'chosen_sent':
                # if chosen_sent, add wizard response to dialog history
                text += '{}\n'.format(wizard_prev_entry['text'])
            text += apprentice_entry['text']

        # next, get label
        wizard_entry = d['dialog'][idx]
        if self.label_type == 'response':
            labels = [wizard_entry['text']]
        else:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            if self.knowledge_separator and title != TOKEN_NOCHOSEN:
                labels = ['{} {} {}'.format(title, TOKEN_KNOWLEDGE, sentence)]
            else:
                labels = ['{} {}'.format(title, sentence)]

        # finally, get label_candidates
        label_cands = ['{} {}'.format(TOKEN_NOCHOSEN, TOKEN_NOCHOSEN)]
        knowledge_str = ''
        for title, passage in knowledge_dict.items():
            for p in passage:
                if self.knowledge_separator:
                    cand = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, p)
                else:
                    cand = '{} {}'.format(title, p)
                knowledge_str += cand + '\n'
                label_cands.append(cand)
        if self.label_type == 'response':
            if 'train' in self.datatype:
                label_cands = []
            else:
                label_cands = wizard_entry.get('candidate_responses', [])

        action = {
            'id': 'WizardDialogKnowledgeTeacher',
            'text': text,
            'labels': labels,
            'chosen_topic': chosen_topic,
            'episode_done': episode_done,
            'label_candidates': label_cands,
        }
        if self.include_knowledge:
            action['knowledge'] = knowledge_str
        if self.include_checked_sentence:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            action['title'] = title
            action['checked_sentence'] = sentence
        return action

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ):
        """
        Custom Evaluations for Wizard of Wikipedia.

        When the label is `chosen_sent`, evaluate whether the model response...
        1) Is the correct document (title)
        2) _contains_ the correct chosen sentence (even if it's not wholly the answer)

        When the label is `response`, we compute F1 of model generation w.r.t checked sentence.

        :param teacher_action:
            The message last sent from this teacher.
        :param labels:
            The previous correct labels, if there were any.
        :param model_response:
            The raw response from the model. Generally you want to rely on the
            text field, but others may be necessary in specific situations.
        """
        if (
            self.label_type == 'response'
            and 'text' in model_response
            and 'checked_sentence' in teacher_action
        ):
            self.metrics.add(
                'knowledge_f1',
                F1Metric.compute(
                    model_response['text'], [teacher_action['checked_sentence']]
                ),
            )
            if labels:
                self.metrics.add(
                    'rare_word_f1',
                    self.rare_word_f1.compute(model_response['text'], labels),
                )
        elif (
            self.label_type == 'chosen_sent'
            and TOKEN_KNOWLEDGE in model_response['text']
        ):
            try:
                correct_title, correct_passage = [
                    normalize_answer(a) for a in labels[0].split(TOKEN_KNOWLEDGE)
                ]
            except ValueError:
                # Knowledge not chosen
                correct_title, correct_passage = TOKEN_NOCHOSEN, TOKEN_NOCHOSEN
            title, passage = [
                normalize_answer(a)
                for a in model_response['text'].split(TOKEN_KNOWLEDGE)
            ]

            self.metrics.add('title_r@1', AverageMetric(int(correct_title == title)))
            self.metrics.add(
                'passage_r@1', AverageMetric(int(correct_passage in passage))
            )
            if 'title_candidates' in model_response:
                title_candidates = [
                    normalize_answer(t) for t in model_response['title_candidates']
                ][:5]
                self.metrics.add(
                    'title_r@5',
                    AverageMetric(
                        int(any(correct_title == t for t in title_candidates))
                    ),
                )
            if 'text_candidates' in model_response:
                text_candidates = [
                    normalize_answer(t) for t in model_response['text_candidates']
                ][:5]
                self.metrics.add(
                    'passage_r@5',
                    AverageMetric(
                        int(any(correct_passage in t for t in text_candidates))
                    ),
                )


class BasicdialogTeacher(WizardOfWikipediaTeacher):
    """
    Teacher that only contains the basic dialog between the wizard and the Apprentice.
    """

    def __init__(self, opt, shared=None):
        self.add_missing_turns = opt.get('add_missing_turns', 'none')
        super().__init__(opt, shared)
        self.speaker_label = opt.get('speaker_label', 'both')
        self.add_topic = opt.get('add_topic', False)
        self.num_exs = sum(self.len_episode(i) for i in range(len(self.data)))

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Basic Dialog Arguments')
        agent.add_argument(
            '--speaker-label',
            type=str,
            default='both',
            choices=['both', 'wizard', 'apprentice'],
            help='Which speaker labels to train on',
        )
        agent.add_argument(
            '--add-topic',
            type='bool',
            default=False,
            help='prepend chosen topic to first turn',
        )
        agent.add_argument(
            '--add-missing-turns',
            type=str,
            choices=['none', 'train', 'all'],
            default='none',
            help='For reproducibility, the default "none" is the previous version which missing some data. '
            'When "train" is chosen, only the training set is supplemented. '
            'When "all" is chosen, all data are supplemented.',
        )
        return parser

    def num_examples(self):
        return self.num_exs

    def len_episode(self, ep):
        d = self.data[ep]
        first_speaker = d['dialog'][0]['speaker'].lower()
        if self.speaker_label != 'both' and self.speaker_label in first_speaker:
            if self.add_missing_turns == 'none':
                warn_once(
                    'Some data not being used. If you are not trying to reproduce '
                    'the previous results, it is recommended that you run with the '
                    'flag --add-missing-turns train or --add-missing-turns all.'
                )
                len_ep = (len(d['dialog']) - 1) // 2
            elif self.add_missing_turns == 'train' and 'train' not in self.datatype:
                len_ep = (len(d['dialog']) - 1) // 2
            else:
                len_ep = (len(d['dialog']) - 1) // 2 + 1
            return len_ep
        return len(d['dialog']) // 2

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        idx = entry_idx * 2
        first_speaker = d['dialog'][0]['speaker'].lower()
        if self.speaker_label != 'both' and self.speaker_label in first_speaker:
            idx += 1

        dialog_entry_1 = d['dialog'][idx]
        dialog_entry_2 = d['dialog'][idx + 1]

        text = dialog_entry_1['text']
        labels = [dialog_entry_2['text']]

        assert isinstance(self.add_topic, bool)
        if self.add_topic and entry_idx == 0:
            text = d.get('chosen_topic', '') + '\n' + text

        action = {
            'id': 'WizardBasicDialog',
            'text': text,
            'labels': labels,
            'episode_done': episode_done,
        }
        if 'label_candidates' in d:
            action['label_candidates'] = d['label_candidates']

        if self.speaker_label == 'wizard':
            action['chosen_topic'] = d.get('chosen_topic', '')

        return action


class BasicWizardDialogTeacher(BasicdialogTeacher):
    def __init__(self, opt, shared=None):
        opt['speaker_label'] = "wizard"
        super().__init__(opt, shared)


class BasicApprenticeDialogTeacher(BasicdialogTeacher):
    def __init__(self, opt, shared=None):
        opt['speaker_label'] = 'apprentice'
        super().__init__(opt, shared)


class BasicBothDialogTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt[
            'task'
        ] = 'wizard_of_wikipedia:BasicApprenticeDialog,wizard_of_wikipedia:BasicWizardDialog'
        super().__init__(opt, shared)


###############################################################
#                                                             #
# Teachers for the Generator                                  #
#                                                             #
###############################################################


class GeneratorTeacher(WizardDialogKnowledgeTeacher):
    """
    Teacher for training a generator.

    Depending on certain flag configurations, the teacher will include differing amounts
    of knowledge
    """

    def __init__(self, opt, shared=None):
        opt['label_type'] = 'response'
        opt['include_checked_sentence'] = True
        super().__init__(opt, shared)
        self.knowledge_separator = opt.get('include_knowledge_separator', True)
        self.only_checked_knowledge = opt.get('only_checked_knowledge', False)
        self.prepend_gold_knowledge = opt.get('prepend_gold_knowledge')
        self.gold_knowledge_delimiter = opt.get('gold_knowledge_delimiter', '\n')
        self.dropout = opt.get('ignorant_dropout', 0.0)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.set_defaults(include_knowledge_separator=True)
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('GeneratorTeacher Arguments')
        agent.add_argument(
            '--only-checked-knowledge',
            type='bool',
            default=False,
            help='If true, only the checked sentence is provided',
        )
        agent.add_argument(
            '--ignorant-dropout',
            type=float,
            default=0.0,
            help='Eliminate all knowledge with this probability.'
            'Specify 1 for completely ignorant teacher',
        )
        agent.add_argument(
            '--prepend-gold-knowledge',
            type='bool',
            default=False,
            help='If true, prepend text with checked sentence',
        )
        agent.add_argument(
            '--gold-knowledge-delimiter',
            type=str,
            default='\n',
            help='delimiter for prepending gold knowledge',
        )
        return parser

    def getID(self):
        return "WizTeacher"

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        # zero out the label candidates?
        if 'knowledge' not in a:
            # just a batch padding item
            return a
        # save some memory, we don't need label_candidates
        a['label_candidates'] = []
        if not a['knowledge'].startswith(TOKEN_NOCHOSEN):
            # make sure the token is appearing
            a['knowledge'] = (
                TOKEN_NOCHOSEN
                + ' '
                + TOKEN_KNOWLEDGE
                + ' '
                + TOKEN_NOCHOSEN
                + '\n'
                + a['knowledge']
            )
        if self.only_checked_knowledge:
            # useful for test time evaluation, where it's only ever trained on true
            # knowledge
            a['knowledge'] = (
                a['title'] + ' ' + TOKEN_KNOWLEDGE + ' ' + a['checked_sentence']
            )

        if random.random() < self.dropout:
            # Drop the knowledge with some probability
            a['title'] = TOKEN_NOCHOSEN
            a['checked_sentence'] = TOKEN_NOCHOSEN
            a['knowledge'] = (
                TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN
            )
        elif self.prepend_gold_knowledge:
            a[
                'text'
            ] = f"{TOKEN_KNOWLEDGE} {a['checked_sentence']} {TOKEN_END_KNOWLEDGE}{self.gold_knowledge_delimiter}{a['text']}"
        return a


class WikiPageTitleTeacher(WizardDialogKnowledgeTeacher):
    """
    Generates the title of Wikipedia page used as source of knowledge.

    The context provided by this teacher (`text`) is the conversation history, with chosen topic removed.
    The label is the title of the Wikipedia page of the passage that wizard selected for crafting
    the next utterance; in other words, the source of knowledge for this utterance.
    """

    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.opt['label_type'] = 'response'
        super().__init__(self.opt, shared=shared)
        self.id = 'WikiPageTitleTeacher'
        self._conv_history_len = self.opt['conversation_history_length']
        if not (self._conv_history_len > 0 or self._conv_history_len == -1):
            logging.warning(
                f'"{self._conv_history_len}" is an invalid value for --conversation-history-length flag.'
                ' Changing it to default of -1 (include the entire message history).'
            )
            self._conv_history_len = -1
        self._skip_no_title = self.opt['skip_no_title']
        if not shared:
            self._preprocess_data()
        else:
            self.titles_data = shared['titles_data']

    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('Wikipedia Page Title Arguments')
        agent.add_argument(
            '--conversation-history-length',
            type=int,
            default=-1,
            help='Number of previous utterances to keep in context, 0 (default) includes all',
        )
        agent.add_argument(
            '--skip-no-title',
            type='bool',
            default=True,
            help=(
                'Whether to skip the example if no passage was selected. If `false` '
                f'uses `{TOKEN_NOCHOSEN}` instead of title if no knowledge source was selected.'
            ),
        )
        return parser

    def share(self):
        shared = super().share()
        shared['titles_data'] = self.titles_data
        return shared

    def _generate_messages(self, hist, action):
        include_hist = (
            hist[-self._conv_history_len :] if self._conv_history_len > 0 else hist
        )
        context = '\n'.join(include_hist)
        return Message(
            {
                'id': "Wikipedia Title Teacher",
                'text': context,
                'labels': [action["title"]],
                'episode_done': True,
            }
        )

    def _should_include(self, act):
        return not (self._skip_no_title and act['labels'][0] == TOKEN_NOCHOSEN)

    def _preprocess_data(self):
        data = []
        for episode_idx in range(super().num_episodes()):
            dialog_history = []
            ex_idx = 0
            while True:
                a = super().get(episode_idx, ex_idx)
                text_parts = a['text'].split('\n')
                if ex_idx == 0:
                    # throwing away chosen_topic
                    text_parts = text_parts[1:]
                if text_parts:
                    dialog_history.append(text_parts[0])
                    title_act = self._generate_messages(dialog_history, a)
                    if self._should_include(title_act):
                        data.append(title_act)
                if a['episode_done']:
                    break
                ex_idx += 1
                dialog_history.append(a['labels'][0])

        logging.info(
            f'{len(data)} title generation examples generated '
            f'from {super().num_examples()} original examples'
        )
        self.titles_data = data

    def num_episodes(self):
        return len(self.titles_data)

    def num_examples(self):
        return self.num_episodes()

    def get(self, episode_idx, entry_idx=0):
        return self.titles_data[episode_idx]


####################################################
#                                                  #
# Doc Reader Teachers                              #
#                                                  #
####################################################


class DocreaderTeacher(WizardOfWikipediaTeacher):
    """
    Teacher for training a doc reader. One can specify the format of the action via the
    `--teacher-type` flag.

    docs:
        {
            text: <Passage> \n <Sentence for which passage was retrieved>
            labels: <Sentence chosen from passage>
        }

    docs_sentence:
        {
            text: <Sentence for which passage was retrieved>
            label: <Sentence chosen from passages>
            label_candidates: <All sentences in retrieved passage>
        }

    more_docs:
        {
            text: <All retrieved passages> \n
                  <Chosen topic + Last thing wizard said + last thing apprentice said>
            labels: <Sentence chosen from passages>
        }

    more_docs_sentence:
        {
            text: <Sentence for which passage was retrieved>
            label: <Sentence chosen from passages>
            label_candidates: <All sentences in all retrieved passages>
        }
    span:
        {
            text: <Sentence for which passage was retrieved>
            label: <Max overlap span between sentence said and sentence retrieved>
        }
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        # get number of examples
        self.num_exs = 0
        for ep in range(self.num_episodes()):
            d = self.data[ep]
            for entry in d['dialog']:
                if (
                    entry.get('checked_sentence', None) is not None
                    and entry.get('checked_sentence') != {}
                    and TOKEN_NOCHOSEN not in entry.get('checked_sentence')
                ):
                    self.num_exs += 1
        self.stop_words = [
            'i',
            'a',
            'an',
            'am',
            'are',
            'about',
            'as',
            'at',
            'be',
            'by',
            'for',
            'from',
            'how',
            'in',
            'is',
            'it',
            'of',
            'on',
            'or',
            'that',
            'the',
            'this',
            'to',
            'was',
            'what',
            'when',
            'where',
            '--',
            '?',
            '.',
            "''",
            "''",
            "``",
            ',',
            'do',
            'see',
            'want',
            'people',
            'and',
            "n't",
            "me",
            'too',
            'own',
            'their',
            '*',
            "'s",
            'not',
            'than',
            'other',
            'you',
            'your',
            'know',
            'just',
            'but',
            'does',
            'really',
            'have',
            'into',
            'more',
            'also',
            'has',
            'any',
            'why',
            'will',
            'with',
            'well',
            'still',
            'he',
            'she',
            'we',
            'may',
            'these',
            'his',
            'hers',
            'which',
            'such',
            'they',
            'its',
            'were',
            'my',
            'there',
            ';',
            '-',
            ':',
            '|',
            '&',
            ')',
            '(',
        ]

        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        # nltk-specific setup
        st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download('punkt')
            self.sent_tok = nltk.data.load(st_path)

        self.teacher_type = opt.get('teacher_type')

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        WizardDialogKnowledgeTeacher.add_cmdline_args(parser, partial_opt=partial_opt)
        parser.add_argument(
            '--teacher-type',
            type=str,
            default='docs',
            help='determines what the action dict looks like; see docstring '
            'for examples',
            choices=[
                'docs',
                'docs_sentence',
                'more_docs',
                'more_docs_sentence',
                'span_teacher',
            ],
        )
        return parser

    def get_min_stopwords(self, word_set):
        min_count = 1000000000000
        min_words = ''
        for words in word_set:
            count = 0
            for stop in self.stop_words:
                if stop in words:
                    count += 1
            if count < min_count:
                min_count = count
                min_words = words
        return min_words

    def space_punctuation(self, words, unspace=False):
        puncs = [
            ('.', ' .'),
            (',', ' ,'),
            ('?', ' ?'),
            (' !', '!'),
            ('(', ' ('),
            (')', ' )'),
        ]
        new_words = words
        for punc in puncs:
            if unspace:
                new_words = new_words.replace(punc[1], punc[0])
            else:
                new_words = new_words.replace(punc[0], punc[1])
        return new_words

    def get_span(self, one, two):
        if not one or not two:
            return None
        one_space = self.space_punctuation(one)
        two_space = self.space_punctuation(two)
        first = one_space.split(' ')
        second = two_space.split(' ')
        length = min(len(first), len(second))
        overlap = set.intersection(set(first), set(second))
        if not overlap:
            return ''
        max_span = self.space_punctuation(self.get_min_stopwords(overlap), unspace=True)
        for i in range(1, length):
            t_1 = []
            t_2 = []
            for j in range(len(first) - i):
                temp_1 = ' '.join([first[k] for k in range(j, j + i + 1)])
                t_1.append(temp_1)
            for j in range(len(second) - i):
                temp_2 = ' '.join([second[k] for k in range(j, j + i + 1)])
                t_2.append(temp_2)
            overlap = set.intersection(set(t_1), set(t_2))
            if not overlap:
                return max_span
            max_span = self.space_punctuation(
                self.get_min_stopwords(overlap), unspace=True
            )
        return max_span

    def num_examples(self):
        return self.num_exs

    def length_episode(self, dialog):
        len_ep = 0
        idxs = []
        i = 0
        for entry in dialog['dialog']:
            if (
                entry.get('checked_sentence', None) is not None
                and entry.get('checked_sentence') != {}
                and TOKEN_NOCHOSEN not in entry.get('checked_sentence')
            ):
                len_ep += 1
                idxs.append(i)
            i += 1

        return len_ep, idxs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        len_ep, idxs = self.length_episode(d)
        idx = idxs[entry_idx]

        episode_done = entry_idx == len_ep - 1
        checked_sentence_dict = d['dialog'][idx]['checked_sentence']

        # get selected sentence
        sentence = _first_val(checked_sentence_dict)

        # get passage of selected topic, text for dialog
        passage, text = self.extract_passage_and_text(d, idx)

        # get all available passages, all texts in previous 3 utterances
        passages, texts = self.extract_passages_and_texts(d, idx)

        # get sentence span
        span_label = self.get_span_label(d, idx)

        action = {
            'id': 'WizardDocReader:{}'.format(self.teacher_type),
            'labels': [sentence],
            'episode_done': episode_done,
        }

        if self.teacher_type == 'docs':
            action['text'] = '{}\n{}'.format(passage, text)
        elif self.teacher_type == 'docs_sentence':
            action['text'] = text
            action['label_candidates'] = self.sent_tok.tokenize(passage)
        elif self.teacher_type == 'more_docs':
            action['text'] = '{}\n{}'.format(passages, texts)
        elif self.teacher_type == 'more_docs_sentence':
            action['text'] = texts
            action['label_candidates'] = self.sent_tok.tokenize(passages)
            label = action['labels'][0]
            if label not in action['label_candidates']:
                action['label_candidates'].append(label)
        elif self.teacher_type == 'span':
            action['text'] = '{}\n{}'.format(passages, texts)
            action['labels'] = [span_label]

        return action

    def extract_passage_and_text(self, data, idx):
        passage_key = _first_key(data['dialog'][idx]['checked_sentence'])
        dialog_entry = data['dialog'][idx]
        text = passage = None
        if 'chosen' in passage_key:
            # from chosen topic
            passage = ' '.join(data['chosen_topic_passage'])
            text = data['chosen_topic']
        elif 'self' in passage_key:
            # from last thing wizard said
            passages = data['dialog'][idx - 2]['retrieved_passages']
            passage = None
            key = _first_val(dialog_entry['checked_passage'])
            for p in passages:
                if key in p:
                    passage = ' '.join(p[key])
                    break
            text = data['dialog'][idx - 2]['text']
        elif 'partner' in passage_key:
            # from last thing partner said
            passages = data['dialog'][idx - 1]['retrieved_passages']
            passage = None
            key = _first_val(dialog_entry['checked_passage'])
            for p in passages:
                if key in p:
                    passage = ' '.join(p[key])
                    break
            text = data['dialog'][idx - 1]['text']

        return passage, text

    def extract_passages_and_texts(self, d, idx):
        # get chosen topic passages and text
        chosen_passages = ' '.join(d['chosen_topic_passage'])
        chosen_text = d['chosen_topic']

        # get apprentice passages and text
        if (idx - 1) >= 0:
            appr_passages = d['dialog'][idx - 1]['retrieved_passages']
            appr_text = d['dialog'][idx - 1]['text']
            appr_list = []
            for passage in appr_passages:
                for v in passage.values():
                    temp = ' '.join(v)
                    appr_list.append(temp)
            appr = '\n'.join(appr_list)
        else:
            appr_passages = ''
            appr_text = ''

        # get wizard passages and text
        if (idx - 2) >= 0:
            wizard_passages = d['dialog'][idx - 2]['retrieved_passages']
            wizard_text = d['dialog'][idx - 2]['text']
            wizard_list = []
            for passage in wizard_passages:
                for v in passage.values():
                    temp = ' '.join(v)
                    wizard_list.append(temp)
            wizard = '\n'.join(wizard_list)
        else:
            wizard_passages = ''
            wizard_text = ''

        if (idx - 2) >= 0:
            passages = '\n'.join([chosen_passages, wizard, appr])
            texts = ' '.join([chosen_text, wizard_text, appr_text])
        elif (idx - 1) >= 0:
            passages = '\n'.join([chosen_passages, appr])
            texts = ' '.join([chosen_text, appr_text])
        else:
            passages = chosen_passages
            texts = chosen_text

        return passages, texts

    def get_span_label(self, data, idx):
        dialog_entry = data['dialog'][idx]
        said = dialog_entry['text']
        sentence = _first_val(dialog_entry['checked_sentence'])
        overlap = self.get_span(said, sentence)
        if not overlap or overlap in self.stop_words:
            label = sentence
        else:
            label = overlap

        return label


class DefaultTeacher(WizardDialogKnowledgeTeacher):
    pass


class SelfchatTeacher(BasicBothDialogTeacher):
    """
    Teacher used to create candidates for selfchats, if needed.
    """

    pass
