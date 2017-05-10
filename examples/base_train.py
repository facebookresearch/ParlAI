# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic template of training loop.
We create an agent that will train on the training task, and be evaluated
on the validation version of the task.

We then do one iteration over ten training examples and one validation example,
printing reports from those tasks after completing those iterations.

This code is meant as a basic template: more advanced loops can iterate over
a validation dataset for exactly one epoch, can take in command-line arguments
using the argument parser in the core library, or generate a dictionary before
processing the data.
"""

from parlai.core.params import ParlaiParser
from parlai.core.agents import Agent
from parlai.core.worlds import create_task
import time

def main():
    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=10)
    opt = parser.parse_args()

    agent = Agent(opt)

    opt['datatype'] = 'train'
    world_train = create_task(opt, agent)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    start = time.time()
    # train / valid loop
    for _ in range(1):
        print('[ training ]')
        for _ in range(10):  # train for a bit
            world_train.parley()

        print('[ training summary. ]')
        print(world_train.report())

        print('[ validating ]')
        for _ in range(1):  # check valid accuracy
            world_valid.parley()

        print('[ validation summary. ]')
        print(world_valid.report())

    print('finished in {} s'.format(round(time.time() - start, 2)))

if __name__ == '__main__':
    main()
