#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build

import copy
import json
import os


def get_sentence_tokenizer():
    """
    Loads the nltk sentence tokenizer.
    """
    try:
        import nltk
    except ImportError:
        raise ImportError('Please install nltk (e.g. pip install nltk).')
    # nltk-specific setup
    st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
    try:
        sent_tok = nltk.data.load(st_path)
    except LookupError:
        nltk.download('punkt')
        sent_tok = nltk.data.load(st_path)
    return sent_tok


class IndexTeacher(FixedDialogTeacher):
    """
    Hand-written SQuAD teacher, which loads the json squad data and implements its own
    `act()` method for interacting with student agent, rather than inheriting from the
    core Dialog Teacher. This code is here as an example of rolling your own without
    inheritance.

    This teacher also provides access to the "answer_start" indices that specify the
    location of the answer in the context.
    """

    def __init__(self, opt, shared=None):
        build(opt)
        super().__init__(opt, shared)

        if self.datatype.startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        datapath = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
        self.data = self._setup_data(datapath)

        self.id = 'squad'
        self.reset()

    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        question = qa['question']
        answers = []
        answer_starts = []
        for a in qa['answers']:
            answers.append(a['text'])
            answer_starts.append(a['answer_start'])
        context = paragraph['context']

        action = {
            'id': 'squad',
            'text': context + '\n' + question,
            'labels': answers,
            'episode_done': True,
            'answer_starts': answer_starts,
        }
        return action

    def _setup_data(self, path):
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        self.examples = []

        for article_idx in range(len(self.squad)):
            article = self.squad[article_idx]
            for paragraph_idx in range(len(article['paragraphs'])):
                paragraph = article['paragraphs'][paragraph_idx]
                num_questions = len(paragraph['qas'])
                for qa_idx in range(num_questions):
                    self.examples.append((article_idx, paragraph_idx, qa_idx))


class DefaultTeacher(DialogTeacher):
    """
    This version of SQuAD inherits from the core Dialog Teacher, which just requires it
    to define an iterator over its data `setup_data` in order to inherit basic metrics,
    a default `act` function.

    For SQuAD, this does not efficiently store the paragraphs in memory.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        opt['datafile'] = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
        self.id = 'squad'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        for article in self.squad:
            # each paragraph is a context for the attached questions
            for paragraph in article['paragraphs']:
                # each question is an example
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = (a['text'] for a in qa['answers'])
                    context = paragraph['context']
                    yield (context + '\n' + question, answers), True


class OpensquadTeacher(DialogTeacher):
    """
    This version of SQuAD inherits from the core Dialog Teacher, which just requires it
    to define an iterator over its data `setup_data` in order to inherit basic metrics,
    a default `act` function.

    Note: This teacher omits the context paragraph
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        opt['datafile'] = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
        self.id = 'squad'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        for article in self.squad:
            # each paragraph is a context for the attached questions
            for paragraph in article['paragraphs']:
                # each question is an example
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = (a['text'] for a in qa['answers'])
                    yield (question, answers), True


class TitleTeacher(DefaultTeacher):
    """
    This version of SquAD inherits from the Default Teacher.

    The only
    difference is that the 'text' field of an observation will contain
    the title of the article separated by a newline from the paragraph and the
    query.
    Note: The title will contain underscores, as it is the part of the link for
    the Wikipedia page; i.e., the article is at the site:
    https://en.wikipedia.org/wiki/{TITLE}
    Depending on your task, you may wish to remove underscores.
    """

    def __init__(self, opt, shared=None):
        self.id = 'squad_title'
        build(opt)
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        for article in self.squad:
            title = article['title']
            # each paragraph is a context for the attached questions
            for paragraph in article['paragraphs']:
                # each question is an example
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = (a['text'] for a in qa['answers'])
                    context = paragraph['context']
                    yield ('\n'.join([title, context, question]), answers), True


class FulldocTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'valid'
        datafile = os.path.join(
            opt['datapath'], 'SQuAD-fulldoc', "squad_fulldocs." + suffix + ":ordered"
        )
        opt['parlaidialogteacher_datafile'] = datafile
        super().__init__(opt, shared)
        self.id = 'squad-fulldoc'
        self.reset()


class SentenceTeacher(IndexTeacher):
    """
    Teacher where the label(s) are the sentences that contain the true answer.

    Some punctuation may be removed from the context and the answer for
    tokenization purposes.

    If `include_context` is False, the teacher returns action dict in the
    following format:
    {
        'context': <context>,
        'text': <question>,
        'labels': <sentences containing the true answer>,
        'label_candidates': <all sentences in the context>,
        'episode_done': True,
        'answer_starts': <index of start of answer in context>
    }
    Otherwise, the 'text' field contains <context>\n<question> and there is
    no separate context field.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.sent_tok = get_sentence_tokenizer()
        self.include_context = opt.get('include_context', False)

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('SQuAD Sentence Teacher Arguments')
        agent.add_argument(
            '--include-context',
            type='bool',
            default=False,
            help='include context within text instead of as a ' 'separate field',
        )

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        context = paragraph['context']
        question = qa['question']

        answers = [a['text'] for a in qa['answers']]

        # remove '.', '?', '!' from answers for proper sentence
        # tokenization
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace('.', '').replace('?', '').replace('!', '')
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)

        edited_sentences = self.sent_tok.tokenize(context)

        labels = []
        label_starts = []
        for sentence in edited_sentences:
            for answer in edited_answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))
                    break

        action = {
            'context': context,
            'text': question,
            'labels': labels,
            'label_candidates': edited_sentences,
            'episode_done': True,
            'answer_starts': label_starts,
        }

        if self.include_context:
            action['text'] = action['context'] + '\n' + action['text']
            del action['context']

        return action


class FulldocsentenceTeacher(FulldocTeacher):
    """
    Teacher which contains the question as the text, the sentences as the label
    candidates, and the label as the sentence containing the answer.

    Some punctuation may be removed for tokenization purposes.

    If `include_context` is False, the teacher returns action dict in the
    following format:
    {
        'context': <context>,
        'text': <question>,
        'labels': <sentences containing the true answer>,
        'label_candidates': <all sentences in the context>,
        'episode_done': True,
        'answer_starts': <index of start of answer in context>
    }
    Otherwise, the 'text' field contains <context>\n<question> and there is
    no separate context field.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.sent_tok = get_sentence_tokenizer()
        self.include_context = opt.get('include_context', False)

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('SQuAD Fulldoc Sentence Teacher Arguments')
        agent.add_argument(
            '--include-context',
            type='bool',
            default=False,
            help='include context within text instead of as a ' 'separate field',
        )

    def get(self, episode_idx, entry_idx=None):
        action = {}
        episode = self.episodes[episode_idx][entry_idx]
        context = ' '.join(episode['text'].split('\n')[:-1]).replace(
            '\xa0', ' '
        )  # get rid of non breaking space characters
        question = episode['text'].split('\n')[-1]
        label_field = 'labels' if 'labels' in episode else 'eval_labels'
        answers = []
        for answer in episode[label_field]:
            new_answer = answer.replace('.', '').replace('?', '').replace('!', '')
            context = context.replace(answer, new_answer)
            answers.append(new_answer)
        sentences = self.sent_tok.tokenize(context)
        labels = []
        label_starts = []
        for sentence in sentences:
            for answer in answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))

        action = {
            'context': context,
            'text': question,
            label_field: labels,
            'answer_starts': label_starts,
            'label_candidates': sentences,
            'episode_done': episode['episode_done'],
        }

        if self.include_context:
            action['text'] = action['context'] + '\n' + action['text']
            del action['context']

        return action
