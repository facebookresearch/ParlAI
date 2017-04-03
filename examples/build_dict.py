#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_task_teacher

# Get command line arguments
argparser = ParlaiParser()
DictionaryAgent.add_cmdline_args(argparser)
opt = argparser.parse_args()

teacher_class = create_task_teacher(opt)

opt['datatype'] = 'train'
opt['ordered'] = True
teacher = teacher_class(opt)
dictionary = DictionaryAgent(opt)
world_train = DialogPartnerWorld(opt, [teacher, dictionary])

# Show some example dialogs:
for _ in range(len(teacher)):
    world_train.parley()

if 'dict_savepath' in opt:
    dictionary.save(opt['dict_savepath'])
world_train.shutdown()
