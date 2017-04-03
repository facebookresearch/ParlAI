#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.


from parlai.agents.repeat_label.agents import RepeatLabelAgent
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.agents import create_task_teacher
from parlai.core.params import ParlaiParser

import random
random.seed(42)

# Get command line arguments
opt = ParlaiParser().parse_args()
teacher = create_task_teacher(opt)
agent = RepeatLabelAgent(opt)
world = DialogPartnerWorld(opt, [teacher, agent])

# Show some example dialogs:
for k in range(1000):
        world.parley()
        world.display()
        if k > 100 and world.done():
            break

world.shutdown()
