#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.


from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.agents import RepeatLabelAgent
from parlai.core.worlds import create_task

import random
random.seed(42)

# Get command line arguments
opt = ParlaiParser().parse_args()
agent = RepeatLabelAgent(opt)
world = create_task(opt, agent)

# Show some example dialogs:
with world:
    for k in range(100):
        world.parley()
        print(world.display() + "\n~~")
        if k > 10 and world.done():
            break
