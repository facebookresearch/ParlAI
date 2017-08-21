# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Simple agent which repeats back the labels sent to it.

By default, replies with a single random label from the list of labels sent to
it, if any. If the ``label_candidates`` field is set, will fill the ``text_candidates``
field with up to a hundred randomly selected candidates (the first text
candidate is the selected label).

Options:

    ``returnOneRandomAnswer`` -- default ``True``, set to ``False`` to instead reply with all labels joined by commas.

    ``cantAnswerPercent`` -- default ``0``, set value in range[0,1] to set chance of replying with "I don't know."
"""

import random

from parlai.core.agents import Agent


class RepeatLabelAgent(Agent):

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.returnOneRandomAnswer = opt.get('returnOneRandomAnswer', True)
        self.cantAnswerPercent = opt.get('cantAnswerPercent', 0)
        self.id = 'RepeatLabelAgent'

    def act(self):
        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to repeat yet.'}
        reply = {}
        reply['id'] = self.getID()
        if ('labels' in obs and obs['labels'] is not None
                and len(obs['labels']) > 0):
            labels = obs['labels']
            if random.random() >= self.cantAnswerPercent:
                if self.returnOneRandomAnswer:
                    reply['text'] = labels[random.randrange(len(labels))]
                else:
                    reply['text'] = ', '.join(labels)
            else:
                # Some 'self.cantAnswerPercent' percentage of the time
                # the agent does not answer.
                reply['text'] = "I don't know."
        else:
            reply['text'] = "I don't know."

        if 'label_candidates' in obs and len(obs['label_candidates']) > 0:
            # Produce text_candidates by selecting random candidate labels.
            reply['text_candidates'] = [reply['text']]
            reply['text_candidates'].extend(random.sample(
                obs['label_candidates'], min(len(obs['label_candidates']), 99)))

        return reply
