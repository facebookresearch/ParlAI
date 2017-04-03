#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.agents import Agent, Teacher
from parlai.core.worlds import DialogPartnerWorld
import time

opt = {}
agent = Agent(opt)

opt['datatype'] = 'train'
teacher_train = Teacher(opt)

opt['datatype'] = 'valid'
teacher_valid = Teacher(opt)

world_train = DialogPartnerWorld(opt, [teacher_train, agent])
world_valid = DialogPartnerWorld(opt, [teacher_valid, agent])

start = time.time()
# train / valid loop
for _ in range(1):
    print('[ training ]')
    for _ in range(3):  # do one epoch of train
        world_train.parley()

    print('[ training summary. ]')
    print(teacher_train.report())

    print('[ validating ]')
    for _ in range(1):  # check valid accuracy
        world_valid.parley()

    print('[ validating summary. ]')
    print(teacher_valid.report())

print('finished in {} s'.format(round(time.time() - start, 2)))
