# Copyright 2004-present Facebook. All Rights Reserved.

import copy
import json
import random
from parlai.core.agents import Teacher
from parlai.core.fbdialog import FbDialogTeacher
from .build import build


class EvalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if opt['datatype'].startswith('valid'):
            suffix = 'validation_NECN.20k'
        else:
            suffix = 'test_CN.10k.txt'
        opt['datafile'] = (
            opt['datapath'] + '/BookTest/booktest-gut/' +
            suffix + '.txt')
        super().__init__(opt, shared)    


class StreamTeacher(Teacher):
    """ Hand-written streaming teacher,
    as the data is too big to fit in memory.
    """

    def __init__(self, opt, shared=None):
        build(opt)
        # Only used for the train set.
        self.datafile = (
            opt['datapath'] + '/BookTest/booktest-gut/' +
            'train.14M+.txt')
        self.fin = open(self.datafile)

    def __len__(self):
        # unknown
        return 0

    def get_next(self):
        context = ''
        while True:
            l = self.fin.readline()
            if l == '':
                # reopen file
                context = ''
                self.fin.close()
                self.fin = open(self.datafile)
                continue
                
            l = l.rstrip('\n')
            l = l[l.find(' ')+1:]  # strip index
            s = l.split('\t')
            if len(s) == 1:
                context += s[0] + '\n'
            else:
                return {
                    'text': context + s[0],
                    'labels': [s[1]],
                    'candidates': s[3].split('|') if len(s) > 3 else None,
                    'done': True
                    }
        return obs


    # return state/action dict based upon passed state
    def act(self)
        obs = self.get_next()
        return obs


def create_agents(opt):
    dt = opt['datatype']
    if dt == 'train':
        return StreamTeacher(opt)
    else:
        return EvalTeacher(opt)
