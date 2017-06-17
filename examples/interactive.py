# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which allows local human keyboard input to talk to a trained model.

For example:
`wget https://s3.amazonaws.com/fair-data/parlai/_models/drqa/squad.mdl`
`python examples/interactive.py -m drqa -mf squad.mdl`

Then enter something like:
"Bob is Blue.\nWhat is Bob?"
as the user input (or in general for the drqa model, enter
a context followed by '\n' followed by a question all as a single input.)
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import random

def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.add_argument('-n', '--num-examples', default=1000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    opt = parser.parse_args()
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    print(opt)
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs:
    for k in range(int(opt['num_examples'])):
        world.parley()
        if opt['display_examples']:
            print("---")
            print(world.display() + "\n~~")
        if world.epoch_done():
            print("EPOCH DONE")
            break
    world.shutdown()

if __name__ == '__main__':
    main()
