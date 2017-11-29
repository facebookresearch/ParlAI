# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import DialogTeacher
from .build import build

import copy
import os


def _format(line):
    return line.strip().replace('##AT##-##AT##', '__AT__')


def _path(opt, source, target):
    build(opt)
    dt = opt['datatype'].split(':')[0]
    base = os.path.join(opt['datapath'], 'iwslt14', dt + '.{}')
    return base.format(source), base.format(target)


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        task = opt.get('task', 'iwslt14:de_en')
        if ':' not in task:
            # default to de_en
            task_name = 'de_en'
        else:
            task_name = task.split(':')[1]
        source, target = task_name.split('_')
        opt['datafile'] = _path(opt, source, target)
        super().__init__(opt, shared)

    def setup_data(self, path):
        source, target = path
        with open(source) as src, open(target) as tgt:
            for s in src:
                s = _format(s)
                t = _format(tgt.readline())
                yield (s, [t]), True


class EnDeTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['task'] = 'iwslt14:en_de'
        super().__init__(opt, shared)


class DeEnTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['task'] = 'iwslt14:de_en'
        super().__init__(opt, shared)
