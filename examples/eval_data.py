# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.agents import RepeatLabelAgent
from parlai.core.worlds import create_task

import random

def main():
    random.seed(42)

    # Get command line arguments
    opt = ParlaiParser().parse_args()
    agent = RepeatLabelAgent(opt)
    agent.cantAnswerPercent = 0.35
    world = create_task(opt, agent)

    # Show some example dialogs:
    for k in range(1000):
        world.parley()

        print("---")
        print(world.report())

        if k > 100 and world.episode_done():
            break

    world.shutdown()

if __name__ == '__main__':
    main()
