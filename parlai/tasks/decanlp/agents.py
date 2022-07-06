#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import MultiTaskTeacher
import parlai.tasks.squad.agents as squad
import parlai.tasks.iwslt14.agents as iwslt14
import parlai.tasks.cnn_dm.agents as cnn_dm
import parlai.tasks.multinli.agents as multinli
import parlai.tasks.sst.agents as sst
import parlai.tasks.qasrl.agents as qasrl
import parlai.tasks.qazre.agents as qazre
import parlai.tasks.woz.agents as woz
import parlai.tasks.wikisql.agents as wikisql
import parlai.tasks.mwsc.agents as mwsc
from copy import deepcopy


class SquadTeacher(squad.DefaultTeacher):
    pass


class Iwslt14Teacher(iwslt14.DefaultTeacher):
    def __init__(self, opt, shared=None):
        # remove the 'decanlp prefix from the task' so the default teacher can parse it
        opt = deepcopy(opt)
        opt['task'] = opt['task'][8:]
        super().__init__(opt, shared)


class CnnDmTeacher(cnn_dm.DefaultTeacher):
    pass


class MultinliTeacher(multinli.DefaultTeacher):
    pass


class SstTeacher(sst.DefaultTeacher):
    pass


class QasrlTeacher(qasrl.DefaultTeacher):
    pass


class QazreTeacher(qazre.DefaultTeacher):
    pass


class WozTeacher(woz.DefaultTeacher):
    pass


class WikisqlTeacher(wikisql.DefaultTeacher):
    pass


class MwscTeacher(mwsc.DefaultTeacher):
    pass


class DecaNLPTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        decanlp_tasks = [
            'squad',
            'iwslt14',
            'cnn_dm',
            'multinli',
            'sst',
            'qasrl',
            'qazre',
            'woz',
            'wikisql',
            'mwsc',
        ]
        opt = deepcopy(opt)
        opt['task'] = ', '.join(decanlp_tasks)
        super().__init__(opt, shared)


class DefaultTeacher(DecaNLPTeacher):
    pass
