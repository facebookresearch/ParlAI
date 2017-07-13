# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import copy
import json
import os

from parlai.core.dialog_teacher import DialogTeacher
from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build


def _path(opt, is_passage=False):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]

    if is_passage:  # for passage selection task
        fname = "%s.passage.txt" % dt
    else:
        fname = "%s.txt" % dt

    return os.path.join(opt['datapath'], 'MS_MARCO', fname)


class PassageTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, is_passage=True)
        super().__init__(opt, shared)


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        self.datatype = opt['datatype']
        opt['datafile'] = _path(opt, is_passage=False)
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            for jline in data_file:
                d_example = json.loads(jline)
                context = [d['passage_text'] for d in d_example['passages']]
                question = d_example['query']
                if self.datatype != 'test':
                    answers = d_example['answers']
                    if not answers:
                        answers = ['NULL']  # empty list of answers will cause exception
                else:
                    answers = ['NULL']
                yield ('\n'.join(context) + '\n' + question, answers), True
