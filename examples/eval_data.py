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
agent.cantAnswerPercent = 0.35
world = create_task(opt, agent)

# Show some example dialogs:
for k in range(1000):
    world.parley()

    print("---")
    print(world.report())
    
    if k > 100 and world.done():
        break

world.shutdown()
