# Copyright 2004-present Facebook. All Rights Reserved.

import json
import random
from parlai.core.agents import Teacher
from parlai.core.dialog import DialogTeacher
from .build import build


class Teacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '-filtered')
        super().__init__(opt, shared)


class TestNETeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        super().__init__(opt, shared)

class DefaultTeacher(Teacher):
    """
    Hand-written streaming teacher,
    as the data is too big to fit in memory.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        suffix = opt['datatype']
        if opt['datatype'].startswith('train'):
            suffix = 'train.14M+'
        else:
            if opt['datatype'].startswith('valid'):
                suffix = 'validation_NECN.20k'
            else:
                suffix = 'test_CN.10k.txt'
        self.datafile = (
            opt['datapath'] + 'BookTest/booktest-gut/' +
            suffix + '.txt')
        self.fin = open(self.datafile)

    def __len__(self):
        # unknown
        return 0

    def get_next(self):
        context = ''
        question = ''
        cands = None
        answers = ''
        while True:
            l = self.fin.readline()
            if l == '':
                # reopen file
                context = ''
                question = ''
                cands = None
                answers = ''
                self.fin.close()
                self.fin = open(self.datafile)
                continue
                
            l = l.rstrip('\n')
            l = l[l.find(' ')+1:]  # strip index
            s = l.split('\t')
            if len(s) > 1:
                question = s[0]
                answers = [ s[1] ]
                if len(s) > 3:
                    cands = s[3].split('|')
                break
            else:
                context = context + s[0] + ' '
        obs = {
            'text': context.rstrip(' ') + ' ' + question,
            'labels': answers,
            'candidates': cands,
            'done': True
            }
        return obs

    # return state/action dict based upon passed state
    def act(self, observation):
        obs = self.get_next()
        return obs
