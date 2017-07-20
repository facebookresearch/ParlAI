"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from parlai.convai.convai_world import ConvAIWorld
from parlai.core.params import ParlaiParser
from parlai.core.agents import Agent
from parlai.core.worlds import display_messages

import random


class ConvAISampleAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = "ConvAISampleAgent"
        self.text = 'Nothing to say yet!'
        self.episode_done = False

    def observe(self, observation):
        self.observation = observation
        self.episode_done = observation['episode_done']

        if self.episode_done:
            self.text = '/end'
        else:
            texts = [
                'I love you!',
                'Wow!',
                'Really?',
                'Nice!',
                'Hi',
                'Hello',
                "This is not very interesting. Let's change the subject of the conversation and talk about cats.",
                '/end']
            self.text = texts[random.randint(0, 7)]
        print("\t"+display_messages([observation]))

    def act(self):
        reply = {
            'id': self.getID(),
            'text': self.text,
            'episode_done': self.episode_done
        }
        print("\t"+display_messages([reply]))
        return reply


def main():
    parser = ParlaiParser(True, True)
    parser.add_argument('-bi', '--bot-id')
    parser.add_argument('-bc', '--bot-capacity', default=-1)
    parser.add_argument('-rbu', '--router-bot-url')
    parser.add_argument('-rbpd', '--router-bot-pull-delay', default=1)
    opt = parser.parse_args()
    print(opt)

    shared = {
        'agents': [
            {
                'class': ConvAISampleAgent,
                'opt': opt
            }
        ]
    }

    world = ConvAIWorld(opt, None, shared)

    while True:
        try:
            world.parley()
        except Exception as e:
            print("Exception: {}".format(e))


if __name__ == '__main__':
    main()
