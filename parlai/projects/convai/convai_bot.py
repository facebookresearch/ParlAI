# Copyright (c) 2017, Neural Networks and Deep Learning lab, MIPT
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.convai.convai_world import ConvAIWorld
from parlai.core.params import ParlaiParser
from parlai.core.agents import Agent
from parlai.core.worlds import display_messages

import random


class ConvAISampleAgent(Agent):

    texts =

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
            self.text = random.choice([
                'I love you!',
                'Wow!',
                'Really?',
                'Nice!',
                'Hi',
                'Hello',
                'This is not very interesting. Let\'s change the subject of the conversation and talk about cats.',
                '/end'])

    def act(self):
        reply = {
            'id': self.getID(),
            'text': self.text,
            'episode_done': self.episode_done
        }
        print('\t' + display_messages([reply]))
        return reply


def main():
    parser = ParlaiParser(True, True)
    parser.add_argument('-bi', '--bot-id', required=True,
                        help='Id of local bot used to communicate with RouterBot')
    parser.add_argument('-bc', '--bot-capacity', type=int, default=-1,
                        help='The maximum number of open dialogs. Use -1 for ' +
                             'unlimited number of open dialogs')
    parser.add_argument('-rbu', '--router-bot-url', required=True,
                        help='Url of RouterBot')
    parser.add_argument('-rbpd', '--router-bot-pull-delay', type=int, default=1,
                        help='Delay before new request to RouterBot: minimum 1 sec')
    opt = parser.parse_args()

    agent = ConvAISampleAgent(opt)
    world = ConvAIWorld(opt, agent)

    while True:
        try:
            world.parley()
        except Exception as e:
            print('Exception: {}'.format(e))


if __name__ == '__main__':
    main()
