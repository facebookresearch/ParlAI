# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Teacher
from parlai.core.dialog_teacher import DialogTeacher
from .build import build

import json
import random
import os


class HandwrittenTeacher(Teacher):
    """Hand-written SQuAD teacher, which loads the json squad data and
    implements its own `act()` method for interacting with student agent,
    rather than inheriting from the core Dialog Teacher. This code is here as
    an example of rolling your own without inheritance.
    """

    def __init__(self, opt, shared=None):
        build(opt)
        super().__init__(opt, shared)

        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        datapath = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
        self.data = self._setup_data(datapath)

        self.id = 'squad'
        self.datatype = opt['datatype']
        self.random = self.datatype == 'train'

        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)
        self.reset()

    def __len__(self):
        return len(self.examples)

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        self.metrics.clear()
        self.lastY = None
        self.episode_idx = self.data_offset - self.step_size
        self.episode_done = True
        self.epochDone = False
        if not self.random and self.data_offset >= len(self.examples):
            # could have bigger batchsize then episodes... so nothing to do
            self.epochDone = True

    def observe(self, observation):
        """Process observation for metrics."""
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    # return state/action dict based upon passed state
    def act(self):
        """Send new dialog message."""
        if self.epochDone:
            return {'id': 'squad', 'episode_done': True}

        num_eps = len(self.examples)

        if self.random:
            # select random episode
            self.episode_idx = random.randrange(num_eps)
        else:
            # select next episode
            self.episode_idx = (self.episode_idx + self.step_size) % num_eps

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

        if not self.random and self.episode_idx + self.step_size >= num_eps:
            # this is used for ordered data to check whether there's more data
            self.epochDone = True

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
        print('loading: ' + path)
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
    inherit basic metrics, a default `act` function, and enables
    Hogwild training with shared memory with no extra work.
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
