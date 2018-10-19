from parlai.core.teachers import DialogTeacher
from .build import build

import parlai.tasks.squad.agents as squad
class SquadTeacher(squad.DefaultTeacher): pass

import parlai.tasks.iwslt14.agents as iwslt14
class Iwslt14Teacher(iwslt14.DefaultTeacher): pass

import parlai.tasks.cnn_dm.agents as cnn_dm
class Cnn_DmTeacher(cnn_dm.DefaultTeacher): pass

import parlai.tasks.multinli.agents as multinli
class MultinliTeacher(multinli.DefaultTeacher): pass

import parlai.tasks.sst.agents as sst
class SstTeacher(sst.DefaultTeacher): pass

import parlai.tasks.qasrl.agents as qasrl
class QasrlTeacher(qasrl.DefaultTeacher): pass

import parlai.tasks.qazre.agents as qazre
class QazreTeacher(qazre.DefaultTeacher): pass

import parlai.tasks.woz.agents as woz
class WozTeacher(woz.DefaultTeacher): pass

import parlai.tasks.wikisql.agents as wikisql
class WikisqlTeacher(wikisql.DefaultTeacher): pass

import parlai.tasks.mwsc.agents as mwsc
class MwscTeacher(mwsc.DefaultTeacher): pass


class DecaNLPTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.task_teachers =[
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
        self.data_paths =[teacher.opt['datafile'] for teacher in self.task_teachers]

        print(self.data_paths)

        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'decanlp'

        opt['datafile'] = self._path(opt)

        self.task_setup_data = [self.task_teachers[i].setup_data(self.data_paths[i]) for i in range(10)]

        super().__init__(opt, shared)

    def _path(self,opt):
        build(opt)
        return opt['datapath']

    def setup_data (self, path):
        while True:
            qa_tasks = [next(task)[0] for task in self.task_setup_data]
            if len(qa_tasks)==10:
                yield qa_tasks[0], True
                for task_example in qa_tasks[1:]:
                    yield task_example,  False
            else:
                break

class DefaultTeacher(DecaNLPTeacher):
    pass
