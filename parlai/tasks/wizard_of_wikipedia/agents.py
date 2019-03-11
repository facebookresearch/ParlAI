#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    A dataset with conversations directly grounded with knowledge
    retrieved from Wikipedia. Contains 201k utterances from 22k
    dialogues spanning over 1300 diverse topics, split into train,
    test, and valid sets. The test and valid sets are split
    into two sets each: one with overlapping topics with the train
    set, and one with unseen topics.

    To access the different valid/test splits (unseen/seen), specify
    the corresponding split (`random_split` for seen, `topic_split`
    for unseen) after the last colon in the task.
    E.g. `wizard_of_wikipedia:WizardDialogKnowledgeTeacher:random_split`
"""

from parlai.core.teachers import FixedDialogTeacher
from .build import build

import json
import os
import random


TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'


def _first_val(dictionary):
    return list(dictionary.values())[0]


def _first_key(dictionary):
    return list(dictionary.keys())[0]


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
            if (cand_title1
                    and cand_title1 in k_dict
                    and sentence in k_dict[cand_title1]):
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


class WizardOfWikipediaTeacher(FixedDialogTeacher):
    """The default teacher; essentially reads the json file and outputs the
       raw data.

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
        task = opt.get('task',
                       'wizard_of_wikipedia:WizardOfWikipedia:random_split')
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
        with open(self.data_path) as f:
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
            'retrieved_passages': dialog_entry[
                'retrieved_passages'
            ],
            'checked_sentence': dialog_entry.get('checked_sentence', None),
            'checked_passage': dialog_entry.get('checked_passage', None),
            'episode_done': episode_done
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
        super().__init__(opt, shared)
        self.label_type = opt.get('label_type', 'response')
        self.include_knowledge = opt.get('include_knowledge', True)
        self.include_checked_sentence = opt.get('include_checked_sentence', False)
        self.knowledge_separator = opt.get('include_knowledge_separator', False)
        self.num_exs = sum(self.len_episode(i) for i in range(len(self.data)))

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Wizard Dialog Knowledge arguments')
        agent.add_argument('--label-type', type=str,
                           choices=['response', 'chosen_sent'],
                           default='response',
                           help='whether to populate label field with the '
                           'wizard response, or the chosen sentence')
        agent.add_argument('--include-knowledge', type='bool',
                           default=True,
                           help='Whether to include the knowledge available to'
                           ' the wizard')
        agent.add_argument('--include-checked-sentence', type='bool',
                           default=True,
                           help='Whether to include the Wizard\'s'
                           'checked sentence')
        agent.add_argument('--include-knowledge-separator', type='bool',
                           default=False,
                           help='include special __knowledge__ token between '
                           'title and passage')

    def len_episode(self, ep):
        d = self.data[ep]
        wizard_first = ('Wizard' in d['dialog'][0]['speaker'])
        if wizard_first:
            return (len(d['dialog']) - 1) // 2
        return len(d['dialog']) // 2

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        wizard_first = ('Wizard' in d['dialog'][0]['speaker'])
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

        knowledge_dict = {
            chosen_topic: chosen_topic_passages,
        }
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
            text = '{}\n{}'.format(chosen_topic, apprentice_entry['text'])
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
            title, sentence = _get_chosen_title_and_sent(wizard_entry,
                                                         knowledge_dict)
            labels = ['{} {}'.format(title, sentence)]

        # finally, get label_candidates
        label_cands = [TOKEN_NOCHOSEN + ' ' + TOKEN_NOCHOSEN]
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
            'label_candidates': label_cands
        }
        if self.include_knowledge:
            action['knowledge'] = knowledge_str
        if self.include_checked_sentence:
            title, sentence = _get_chosen_title_and_sent(wizard_entry,
                                                         knowledge_dict)
            action['title'] = title
            action['checked_sentence'] = sentence
        return action


class BasicdialogTeacher(WizardOfWikipediaTeacher):
    """Teacher that only contains the basic dialog between the wizard and
    the Apprentice
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.num_exs = sum(len(d['dialog']) // 2 for d in self.data)
        self.wizard_dialog = opt.get('wizard_dialog', False)

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Basic Dialog Arguments')
        agent.add_argument('--wizard-dialog', type='bool',
                           default=False,
                           help='If true, ensures that wizard response '
                           'is always the label')

    def num_examples(self):
        return self.num_exs

    def len_episode(self, ep):
        d = self.data[ep]
        if self.wizard_dialog and ('Wizard' in d['dialog'][0]['speaker']):
            return (len(d['dialog']) - 1) // 2
        return len(d['dialog']) // 2

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        idx = entry_idx * 2
        if self.wizard_dialog and ('Wizard' in d['dialog'][0]['speaker']):
            idx += 1

        dialog_entry_1 = d['dialog'][idx]
        dialog_entry_2 = d['dialog'][idx + 1]

        text = dialog_entry_1['text']
        labels = [dialog_entry_2['text']]

        action = {
            'id': 'WizardBasicDialog',
            'text': text,
            'labels': labels,
            'episode_done': episode_done,
        }

        if self.wizard_dialog:
            action['chosen_topic'] = d.get('chosen_topic', '')

        return action


###############################################################
#                                                             #
# Teachers for the Generator                                  #
#                                                             #
###############################################################

class GeneratorTeacher(WizardDialogKnowledgeTeacher):
    """Teacher for training a generator. Depending on certain flag
    configurations, the teacher will include differing amounts of knowledge

    """
    def __init__(self, opt, shared=None):
        opt['label_type'] = 'response'
        opt['include_checked_sentence'] = True
        super().__init__(opt, shared)
        self.knowledge_separator = opt.get('include_knowledge_separator', True)
        self.only_checked_knowledge = opt.get('only_checked_knowledge', False)
        self.dropout = opt.get('ignorant_dropout', 0.0)

    @staticmethod
    def add_cmdline_args(argparser):
        argparser.set_defaults(
            include_knowledge_separator=True,
        )
        WizardDialogKnowledgeTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('GeneratorTeacher Arguments')
        agent.add_argument('--only-checked-knowledge', type='bool',
                           default=False,
                           help='If true, only the checked sentence is provided'
                           )
        agent.add_argument('--ignorant-dropout', type=float, default=0.0,
                           help='Eliminate all knowledge with this probability.'
                           'Specify 1 for completely ignorant teacher')

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
                TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN +
                '\n' + a['knowledge']
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
        return a


####################################################
#                                                  #
# Doc Reader Teachers                              #
#                                                  #
####################################################


class DocreaderTeacher(WizardOfWikipediaTeacher):
    """
    Teacher for training a doc reader. One can specify the format of the
    action via the `--teacher-type` flag.

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
                if (entry.get('checked_sentence', None) is not None and
                        entry.get('checked_sentence') != {} and
                        TOKEN_NOCHOSEN not in
                        entry.get('checked_sentence')):
                    self.num_exs += 1
        self.stop_words = ['i', 'a', 'an', 'am', 'are', 'about', 'as', 'at',
                           'be', 'by', 'for', 'from', 'how', 'in', 'is', 'it',
                           'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was',
                           'what', 'when', 'where', '--', '?', '.', "''", "''",
                           "``", ',', 'do', 'see', 'want', 'people', 'and',
                           "n't", "me", 'too', 'own', 'their', '*', "'s", 'not',
                           'than', 'other', 'you', 'your', 'know', 'just',
                           'but', 'does', 'really', 'have', 'into', 'more',
                           'also', 'has', 'any', 'why', 'will', 'with', 'well',
                           'still', 'he', 'she', 'we', 'may', 'these', 'his',
                           'hers', 'which', 'such', 'they', 'its', 'were', 'my',
                           'there', ';', '-', ':', '|', '&', ')', '(']

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

    @staticmethod
    def add_cmdline_args(argparser):
        WizardDialogKnowledgeTeacher.add_cmdline_args(argparser)
        argparser.add_argument(
            '--teacher-type', type=str, default='docs',
            help='determines what the action dict looks like; see docstring '
            'for examples',
            choices=['docs', 'docs_sentence', 'more_docs', 'more_docs_sentence',
                     'span_teacher']
        )

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
        puncs = [('.', ' .'), (',', ' ,'), ('?', ' ?'), (' !', '!'),
                 ('(', ' ('), (')', ' )')]
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
        max_span = self.space_punctuation(self.get_min_stopwords(overlap),
                                          unspace=True)
        for i in range(1, length):
            t_1 = []
            t_2 = []
            for j in range(len(first) - i):
                temp_1 = ' '.join([first[k] for k in range(j, j+i+1)])
                t_1.append(temp_1)
            for j in range(len(second) - i):
                temp_2 = ' '.join([second[k] for k in range(j, j+i+1)])
                t_2.append(temp_2)
            overlap = set.intersection(set(t_1), set(t_2))
            if not overlap:
                return max_span
            max_span = self.space_punctuation(self.get_min_stopwords(overlap),
                                              unspace=True)
        return max_span

    def num_examples(self):
        return self.num_exs

    def length_episode(self, dialog):
        len_ep = 0
        idxs = []
        i = 0
        for entry in dialog['dialog']:
            if (entry.get('checked_sentence', None) is not None and
                    entry.get('checked_sentence') != {} and
                    TOKEN_NOCHOSEN not in entry.get('checked_sentence')):
                len_ep += 1
                idxs.append(i)
            i += 1

        return len_ep, idxs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        len_ep, idxs = self.length_episode(d)
        idx = idxs[entry_idx]

        episode_done = (entry_idx == len_ep - 1)
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
            'episode_done': episode_done
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
