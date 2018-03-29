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
from parlai.core.utils import Timer

import random

def setup_args():
    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.set_defaults(datatype='valid')
    return parser

def eval_model(parser, printargs=True):
    random.seed(42)
    opt = parser.parse_args(print_args=False)
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)
    # Show arguments after loading model
    parser.opt = agent.opt
    if (printargs):
        parser.print_args()
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = Timer()
    tot_time = 0

    # Show some example dialogs:
    for _ in range(int(opt['num_examples'])):
        world.parley()
        if opt['display_examples']:
            print("---")
            print(world.display() + "\n~~")
        if log_time.time() > log_every_n_secs:
            tot_time += log_time.time()
            print(str(int(tot_time)) + "s elapsed: " + str(world.report()))
            log_time.reset()
        if world.epoch_done():
            print("EPOCH DONE")
            break
    print(world.report())


def main():
    eval_model(setup_args())

if __name__ == '__main__':
    main()
