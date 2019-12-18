#!/usr/bin/env python3

# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from projects.convai.convai_world import ConvAIWorld
from parlai.core.params import ParlaiParser
from parlai.core.agents import Agent, create_agent
from parlai.utils.misc import display_messages

import random


class ConvAISampleAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'ConvAISampleAgent'
        self.text = 'Nothing to say yet!'
        self.episode_done = False

    def observe(self, observation):
        print('\t' + display_messages([observation]))
        self.observation = observation
        self.episode_done = observation['episode_done']

        if self.episode_done:
            self.text = '/end'
        else:
            self.text = random.choice(
                [
                    'I love you!',
                    'Wow!',
                    'Really?',
                    'Nice!',
                    'Hi',
                    'Hello',
                    'This is not very interesting. Let\'s change the subject of the '
                    'conversation and talk about cats.',
                    '/end',
                ]
            )

    def act(self):
        reply = {
            'id': self.getID(),
            'text': self.text,
            'episode_done': self.episode_done,
        }
        print('\t' + display_messages([reply]))
        return reply


def setup_args():
    parser = ParlaiParser(True, True)
    ConvAIWorld.add_cmdline_args(parser)
    return parser


def run_convai_bot(opt):
    agent = create_agent(opt)
    world = ConvAIWorld(opt, [agent])
    while True:
        try:
            world.parley()
        except Exception as e:
            print('Exception: {}'.format(e))


def main():
    parser = setup_args()
    parser.set_params(model='projects.convai.convai_bot:ConvAISampleAgent')
    opt = parser.parse_args()
    print('Run ConvAI bot in inifinite loop...')
    run_convai_bot(opt)


if __name__ == '__main__':
    main()
