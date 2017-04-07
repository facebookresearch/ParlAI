#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import unittest
import subprocess
import sys

def check(opt, reply):
    check_no_labels(opt, reply)
    if opt['datatype'].startswith('train'):
        assert reply.get('labels')

def check_no_labels(opt, reply):
    assert reply
    assert reply.get('text')
    assert 'done' in reply
    if 'datafile' in opt:
        # do partial cleaning as we go if possible to save disk space
        subprocess.Popen(['rm', '-rf', opt['datafile']]).communicate()

class TestData(unittest.TestCase):
    """Test access to different datasets."""

    # args to pass to argparse for this test
    TMP_PATH = '/tmp/parlai_test_data/'
    args = [
        '--datatype', 'train',
        '--datapath', TMP_PATH
    ]

    def test_babi_1k(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.babi.agents import Task1kTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for i in range(1, 21):
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'babi:Task1k:{}'.format(i)
                teacher = Task1kTeacher(opt)
                reply = teacher.act({})
                check(opt, reply)


    def test_babi_10k(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.babi.agents import Task10kTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for i in range(1, 21):
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'babi:Task10k:{}'.format(i)
                teacher = Task10kTeacher(opt)
                reply = teacher.act({})
                check(opt, reply)

    def test_cbt(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.booktest.agents import EvalTeacher, StreamTeacher

        opt = ParlaiParser().parse_args(args=self.args)

        opt['datatype'] = 'train:ordered'
        teacher = StreamTeacher(opt)
        reply = teacher.act({})
        check(opt, reply)

        for dt in ['valid', 'test']:
            opt['datatype'] = dt
            teacher = EvalTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)

    def test_cbt(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.cbt.agents import (NETeacher, CNTeacher, VTeacher,
                                               PTeacher)

        opt = ParlaiParser().parse_args(args=self.args)
        for teacher_class in (NETeacher, CNTeacher, VTeacher, PTeacher):
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                teacher = teacher_class(opt)
                reply = teacher.act({})
                check(opt, reply)

    def test_dbll_babi(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.dbll_babi.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)
        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'dbll_babi:task:{}_p{}'.format(i, 0.5)
                teacher = TaskTeacher(opt)
                reply = teacher.act({})
                check(opt, reply)

    def test_dbll_movie(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.dbll_movie.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)

        from parlai.tasks.dbll_movie.agents import KBTeacher
        teacher = KBTeacher(opt)
        reply = teacher.act({})
        check_no_labels(opt, reply)

        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'dbll_movie:task:{}_p{}'.format(i, 0.5)
                teacher = TaskTeacher(opt)
                reply = teacher.act({})
                check(opt, reply)

    def test_dialog_babi(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.dialog_babi.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)

        from parlai.tasks.dialog_babi.agents import KBTeacher
        teacher = KBTeacher(opt)
        reply = teacher.act({})
        check_no_labels(opt, reply)

        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'dialog_babi:task:{}'.format(i)
                teacher = TaskTeacher(opt)
                reply = teacher.act({})
                check(opt, reply)


    def test_mctest(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.mctest.agents import Task160Teacher, Task500Teacher

        opt = ParlaiParser().parse_args(args=self.args)

        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = Task160Teacher(opt)
            reply = teacher.act({})
            check(opt, reply)

            teacher = Task500Teacher(opt)
            reply = teacher.act({})
            check(opt, reply)

    def test_moviedialog(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.moviedialog.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)

        from parlai.tasks.moviedialog.agents import KBTeacher
        teacher = KBTeacher(opt)
        reply = teacher.act({})
        check_no_labels(opt, reply)

        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'moviedialog:task:{}'.format(i)
                teacher = TaskTeacher(opt)
                reply = teacher.act({})
                check(opt, reply)

    def test_mturkwikimovies(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.mturkwikimovies.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)

    def test_qacnn(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.qacnn.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)

    def test_qadailymail(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.qadailymail.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)

    def test_simplequestions(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.simplequestions.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)
            check(opt, reply)

    def test_squad(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.squad.agents import (DefaultTeacher,
                                                 InheritedSquadTeacher)

        opt = ParlaiParser().parse_args(args=self.args)

        for dt in ['train:ordered', 'valid']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)

            teacher = InheritedSquadTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)

    def test_webquestions(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.webquestions.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act({})


    def test_wikimovies(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.wikimovies.agents import DefaultTeacher, KBTeacher

        opt = ParlaiParser().parse_args(args=self.args)

        teacher = KBTeacher(opt)
        reply = teacher.act({})
        check_no_labels(opt, reply)

        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)

    def test_wikiqa(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.wikiqa.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act({})
            check(opt, reply)


if __name__ == '__main__':
    # clean out temp dir first
    subprocess.Popen(['rm', '-rf', TestData.TMP_PATH]).communicate()

    tp = unittest.main(exit=False)
    error_code = len(tp.result.errors)
    if error_code == 0:
        # tests succeeded, clean up
        print('Cleaning up tmp directory, please wait.')
        subprocess.Popen(['rm', '-rf', TestData.TMP_PATH]).communicate()
    else:
        print('At least one test failed. Leaving directory ' +
              '{} with temporary files in place '.format(TestData.TMP_PATH) +
              'for inspection (only failed tasks remain).')
    sys.exit(error_code)
