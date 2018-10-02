#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and runs the given
model on them.

For example:
`python examples/display_model.py -t babi:task1k:1 -m "repeat_label"`
or:
`python examples/display_model.py -t "#MovieDD-Reddit" -m "ir_baseline" -mp "-lp 0.5" -dt test`
"""  # noqa: E501

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import random


def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser(True, True, 'Display model predictions.')
    parser.add_argument('-n', '--num-examples', default=10)
    parser.add_argument('--display-ignore-fields', type=str, default='')
    # by default we want to display info about the validation set
    parser.set_defaults(datatype='valid')
    opt = parser.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    with world:
        for k in range(int(opt['num_examples'])):
            world.parley()
            print(world.display() + "\n~~")
            if world.epoch_done():
                print("EPOCH DONE")
                break


if __name__ == '__main__':
    main()
