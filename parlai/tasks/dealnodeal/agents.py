#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import Teacher
from .build import build

import os
import random

WELCOME_MESSAGE = (
    'Negotiate with your opponent to decide who gets how many of each item. '
    'You must both agree on the distribution of items (or you both get zero), '
    'but try to get as much value as you can. There are {book_cnt} book(s), '
    'each with a value of {book_val}, {hat_cnt} hat(s), each with a value of '
    '{hat_val}, and {ball_cnt} ball(s), each with a value of {ball_val}.'
)

EOS_TOKEN = '<eos>'
SELECTION_TOKEN = '<selection>'
YOU_TOKEN = 'YOU:'
THEM_TOKEN = 'THEM:'

INPUT_TAG = 'input'
DIALOGUE_TAG = 'dialogue'
OUTPUT_TAG = 'output'


def get_tag(tokens, tag):
    """
    Extracts the value inside the given tag.
    """
    start = tokens.index('<' + tag + '>') + 1
    stop = tokens.index('</' + tag + '>')
    return tokens[start:stop]


class NegotiationTeacher(Teacher):
    """
    End-to-end negotiation teacher that loads the data from
    https://github.com/facebookresearch/end-to-end-negotiator.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.datatype = opt['datatype'].split(':')[0]
        self.datatype_ = opt['datatype']
        self.random = self.datatype_ == 'train'
        build(opt)

        filename = 'val' if self.datatype == 'valid' else self.datatype
        data_path = os.path.join(
            opt['datapath'],
            'negotiation',
            'end-to-end-negotiator-master',
            'src',
            'data',
            'negotiate',
            filename + '.txt',
        )

        if shared and 'data' in shared:
            self.episodes = shared['episodes']
        else:
            self._setup_data(data_path)

        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)

        self.reset()

    def num_examples(self):
        # 1 example for every expected learner text response (YOU), and 1
        # example for the expected learner final negotiation output values
        num_exs = 0
        dialogues = [
            self._split_dialogue(get_tag(episode.strip().split(), DIALOGUE_TAG))
            for episode in self.episodes
        ]
        num_exs = sum(
            len([d for d in dialogue if YOU_TOKEN in d]) + 1 for dialogue in dialogues
        )
        return num_exs

    def num_episodes(self):
        return len(self.episodes)

    def reset(self):
        super().reset()
        self.episode_idx = self.data_offset - self.step_size
        self.dialogue_idx = None
        self.expected_reponse = None
        self.epochDone = False

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.episodes = data_file.readlines()

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        if self.expected_reponse is not None:
            self.metrics.evaluate_response(observation, self.expected_reponse)
            self.expected_reponse = None
        return observation

    def act(self):
        if self.dialogue_idx is not None:
            # continue existing conversation
            return self._continue_dialogue()
        elif self.random:
            # if random, then select the next random example
            self.episode_idx = random.randrange(len(self.episodes))
            return self._start_dialogue()
        elif self.episode_idx + self.step_size >= len(self.episodes):
            # end of examples
            self.epochDone = True
            return {'episode_done': True}
        else:
            # get next non-random example
            self.episode_idx = (self.episode_idx + self.step_size) % len(self.episodes)
            return self._start_dialogue()

    def _split_dialogue(self, words, separator=EOS_TOKEN):
        sentences = []
        start = 0
        for stop in range(len(words)):
            if words[stop] == separator:
                sentences.append(words[start:stop])
                start = stop + 1
        if stop >= start:
            sentences.append(words[start:])
        return sentences

    def _start_dialogue(self):
        words = self.episodes[self.episode_idx].strip().split()
        self.values = get_tag(words, INPUT_TAG)
        self.dialogue = self._split_dialogue(get_tag(words, DIALOGUE_TAG))
        self.output = get_tag(words, OUTPUT_TAG)
        # The dialogue should end with a selection token
        assert self.dialogue[-1][1] == SELECTION_TOKEN

        (book_cnt, book_val, hat_cnt, hat_val, ball_cnt, ball_val) = self.values
        welcome = WELCOME_MESSAGE.format(
            book_cnt=book_cnt,
            book_val=book_val,
            hat_cnt=hat_cnt,
            hat_val=hat_val,
            ball_cnt=ball_cnt,
            ball_val=ball_val,
        )

        self.dialogue_idx = -1
        if self.dialogue[0][0] == THEM_TOKEN:
            action = self._continue_dialogue()
            action['text'] = welcome + '\n' + action['text']
        else:
            action = self._continue_dialogue(skip_teacher=True)
            action['text'] = welcome

        action['items'] = {
            "book_cnt": book_cnt,
            "book_val": book_val,
            "hat_cnt": hat_cnt,
            "hat_val": hat_val,
            "ball_cnt": ball_cnt,
            "ball_val": ball_val,
        }

        return action

    def _continue_dialogue(self, skip_teacher=False):
        action = {}

        # Fill in teacher's message (THEM)
        if not skip_teacher:
            self.dialogue_idx += 1
            if self.dialogue_idx >= len(self.dialogue):
                action['text'] = SELECTION_TOKEN
            else:
                sentence = self.dialogue[self.dialogue_idx]
                assert sentence[0] == THEM_TOKEN
                action['text'] = ' '.join(sentence[1:])

        # Fill in learner's response (YOU)
        self.dialogue_idx += 1
        if self.datatype.startswith('train'):
            if self.dialogue_idx >= len(self.dialogue):
                # the agent should finish by reporting what they think the
                # agreed decision was
                self.expected_reponse = [' '.join(self.output)]
            else:
                sentence = self.dialogue[self.dialogue_idx]
                assert sentence[0] == YOU_TOKEN
                self.expected_reponse = [' '.join(sentence[1:])]
            action['labels'] = self.expected_reponse

        if self.dialogue_idx >= len(self.dialogue):
            self.dialogue_idx = None
            action['episode_done'] = True
        else:
            action['episode_done'] = False

        return action


class DefaultTeacher(NegotiationTeacher):
    pass
