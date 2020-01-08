#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Simple agent which repeats back the labels sent to it.

By default, replies with a single random label from the list of labels sent to
it, if any. If the ``label_candidates`` field is set, will fill the ``text_candidates``
field with up to a hundred randomly selected candidates (the first text
candidate is the selected label).

Options:

    ``returnOneRandomAnswer`` -- default ``True``, set to ``False`` to instead
    reply with all labels joined by commas.

    ``cantAnswerPercent`` -- default ``0``, set value in range[0,1] to set
    chance of replying with "I don't know."
"""

import random

from parlai.core.agents import Agent
from parlai.core.message import Message


class RepeatLabelAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('RepeatLabel Arguments')
        group.add_argument(
            '--return_one_random_answer',
            type='bool',
            default=True,
            help='return one answer from the set of labels',
        )
        group.add_argument(
            '--cant_answer_percent',
            type=float,
            default=0,
            help='set value in range[0,1] to set chance of '
            'replying with special message',
        )
        group.add_argument(
            '--cant_answer_message',
            type=str,
            default="I don't know.",
            help='Message sent when the model cannot answer',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.returnOneRandomAnswer = opt.get('return_one_random_answer', True)
        self.cantAnswerPercent = opt.get('cant_answer_percent', 0)
        self.cantAnswerMessage = opt.get('cant_answer_message', "I don't know.")
        self.id = 'RepeatLabelAgent'

    def act(self):
        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to repeat yet.'}
        reply = {}
        reply['id'] = self.getID()
        labels = obs.get('labels', obs.get('eval_labels', None))
        if labels:
            if random.random() >= self.cantAnswerPercent:
                if self.returnOneRandomAnswer:
                    reply['text'] = labels[random.randrange(len(labels))]
                else:
                    reply['text'] = ', '.join(labels)
            else:
                # Some 'self.cantAnswerPercent' percentage of the time
                # the agent does not answer.
                reply['text'] = self.cantAnswerMessage
        else:
            reply['text'] = self.cantAnswerMessage
        reply['episode_done'] = False
        return Message(reply)
