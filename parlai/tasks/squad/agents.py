# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fixed_data_teacher import FixedDataTeacher
from parlai.core.dialog_teacher import DialogTeacher
from .build import build

import json
import os


class IndexTeacher(FixedDataTeacher):
    """Hand-written SQuAD teacher, which loads the json squad data and
    implements its own `act()` method for interacting with student agent,
    rather than inheriting from the core Dialog Teacher. This code is here as
    an example of rolling your own without inheritance.

    This teacher also provides access to the "answer_start" indices that
    specify the location of the answer in the context.
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

    # return state/action dict based upon passed state
    def act(self):
        """Send new dialog message."""
        if self.epochDone and self.training:
            return {'id': 'squad', 'episode_done': True}

        self.episode_idx, self.epochDone = self.next_episode_idx()

        article_idx, paragraph_idx, qa_idx = self.examples[self.episode_idx]
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
            'episode_done': True,
            'answer_starts': answer_starts
        }
        self.lastY = answers

        if 'train' in self.datatype:
            # include labels for training
            action['labels'] = answers
        else:
            # put labels in separate field during eval so we don't accidentally
            # use them during training
            action['eval_labels'] = answers

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
    """This version of SQuAD inherits from the core Dialog Teacher, which just
    requires it to define an iterator over its data `setup_data` in order to
    inherit basic metrics, a default `act` function.
    For SQuAD, this does not efficiently store the paragraphs in memory.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        opt['datafile'] = os.path.join(opt['datapath'], 'SQuAD',
                                       suffix + '-v1.1.json')
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
