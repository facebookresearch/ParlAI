# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and prints them out.
Used for verification of data loading and iteration.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:
`python examples/display_data.py -t babi:task1k:1`
"""

from parlai.core.params import ParlaiParser
from parlai.core.agents import Agent
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import random


class RepeatTextAgent(Agent):
    """Simple agent which repeats the text it receives."""
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RepeatTextAgent'

    def act(self):
        obs = self.observation
        if obs is None or 'text' not in obs:
            return {'text': 'Nothing to repeat yet.'}
        return {'id': self.getID(), 'text': obs['text']}


def display_data(opt):
    # create one repeat label agent and assign it to the specified task
    agents = []
    agents.append(RepeatLabelAgent(opt))

    # since RepeatLabelAgent action has label in 'text', other agents
    # should be `RepeatTextAgent`s
    for _ in range(opt['num_agents'] - 1):
        agents.append(RepeatTextAgent(opt))

    # world will be `DialogPartnerWorld` for num_agents = 1
    # (agent + task teacher), else it will be `MultiAgentDialogWorld`
    world = create_task(opt, agents)

    # Show some example dialogs.
    with world:
        for _ in range(opt['num_examples']):
            world.parley()
            print(world.display() + '\n~~')
            if world.epoch_done():
                print('EPOCH DONE')
                break


def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=10, type=int)
    parser.add_argument('-a', '--num-agents', default=1, type=int)
    opt = parser.parse_args()

    display_data(opt)

if __name__ == '__main__':
    main()
