#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import DialogTeacher
from .build import build
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


class SquadTeacher(squad.DefaultTeacher):
    pass


class Iwslt14Teacher(iwslt14.DefaultTeacher):
    pass


class Cnn_DmTeacher(cnn_dm.DefaultTeacher):
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


class DecaNLPTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.task_teachers = [
            SquadTeacher(opt),
            Iwslt14Teacher(opt),
            Cnn_DmTeacher(opt),
            MultinliTeacher(opt),
            SstTeacher(opt),
            QasrlTeacher(opt),
            QazreTeacher(opt),
            WozTeacher(opt),
            WikisqlTeacher(opt),
            MwscTeacher(opt)
        ]
        self.data_paths = [teacher.opt['datafile'] for teacher in self.task_teachers]

        print(self.data_paths)

        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'decanlp'

        opt['datafile'] = self._path(opt)

        self.task_setup_data = [self.task_teachers[i].setup_data(self.data_paths[i])
                                for i in range(10)]

        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)
        return opt['datapath']

    def setup_data(self, path):
        while True:
            qa_tasks = [next(task)[0] for task in self.task_setup_data]
            if len(qa_tasks) == 10:
                yield qa_tasks[0], True
                for task_example in qa_tasks[1:]:
                    yield task_example, False
            else:
                break


class DefaultTeacher(DecaNLPTeacher):
    pass
