#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.agents.remote_agent.agents import ParsedRemoteAgent
from parlai.core.agents import create_task_teacher
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import DialogPartnerWorld, HogwildWorld

import copy
import time

# Get command line arguments
argparser = ParlaiParser()
DictionaryAgent.add_cmdline_args(argparser)
ParsedRemoteAgent.add_cmdline_args(argparser)
opt = argparser.parse_args()

# set up dictionary
print('Setting up dictionary.')
dictionary = DictionaryAgent(opt)
if not opt.get('dict_loadpath'):
    # build dictionary since we didn't load it
    ordered_opt = copy.deepcopy(opt)
    for datatype in ['train', 'valid']:
        # we use train and valid sets to build dictionary
        ordered_opt['datatype'] = datatype
        ordered_opt['ordered'] = True
        teacher_dict = create_task_teacher(ordered_opt)
        world_dict = DialogPartnerWorld(ordered_opt, [teacher_dict, dictionary])

        # pass examples to dictionary
        for _ in range(len(teacher_dict)):
            world_dict.parley()

    # we need to save the dictionary to load it in memnn (sort it by frequency)
    dictionary.save('/tmp/dict.txt', sort=True)

print('Dictionary ready, moving on to training.')

opt['datatype'] = 'train'
teacher_train = create_task_teacher(opt)
agent = ParsedRemoteAgent(opt, {'dictionary': dictionary})
opt['datatype'] = 'valid'
teacher_valid = create_task_teacher(opt)

world_train = (HogwildWorld(opt, [teacher_train, agent])
               if opt.get('numthreads', 1) > 1 else
               DialogPartnerWorld(opt, [teacher_train, agent]))
world_valid = DialogPartnerWorld(opt, [teacher_valid, agent])

start = time.time()
with world_valid, world_train:
    for _ in range(10):
        print('[ training ]')
        for _ in range(len(teacher_train) * opt.get('numthreads', 1)):
            world_train.parley()
        world_train.synchronize()

        print('[ training summary. ]')
        print(teacher_train.report())

        print('[ validating ]')
        for _ in range(len(teacher_valid)):  # check valid accuracy
            world_valid.parley()

        print('[ validating summary. ]')
        report_valid = teacher_valid.report()
        print(report_valid)
        if report_valid['accuracy'] > 0.95:
            break

    # show some example dialogs after training:
    for _k in range(3):
            world_valid.parley()
            print(world_valid.query['text'])
            print("A: " + world_valid.reply['text'])


print('finished in {} s'.format(round(time.time() - start, 2)))
