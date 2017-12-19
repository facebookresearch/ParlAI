# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and
evaluates the given model on them.

For example:
`python examples/eval_model.py -t "babi:Task1k:2" -m "repeat_label"`
or
`python examples/eval_model.py -t "#CornellMovie" -m "ir_baseline" -mp "-lp 0.5"`
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import random


def eval_model(opt, parser, printargs=True):
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)
    # Show arguments after loading model
    parser.opt = agent.opt
    if (printargs):
        parser.print_args()

    # Show some example dialogs:
    for _ in range(int(opt['num_examples'])):
        world.parley()
        print("---")
        if opt['display_examples']:
            print(world.display() + "\n~~")
        print(world.report())
        if world.epoch_done():
            print("EPOCH DONE")
            break
    world.shutdown()

def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(datatype='valid')
    opt = parser.parse_args(print_args=False)

    eval_model(opt, parser)

if __name__ == '__main__':
    main()
