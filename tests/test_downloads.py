# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import unittest
import shutil
import sys


def check(opt, reply):
    check_no_labels(opt, reply)
    if opt['datatype'].startswith('train'):
        assert reply.get('labels')


def check_no_labels(opt, reply):
    assert reply
    assert reply.get('text')
    assert 'episode_done' in reply


class TestData(unittest.TestCase):
    """Test access to different datasets."""

    # args to pass to argparse for this test
    TMP_PATH = '/tmp/parlai_test_data/'
    args = [
        '--datatype', 'train',
        '--datapath', TMP_PATH
    ]

    def test_babi(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.babi.agents import Task1kTeacher, Task10kTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for i in range(1, 21):
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'babi:Task1k:{}'.format(i)
                teacher = Task1kTeacher(opt)
                reply = teacher.act()
                check(opt, reply)

        for i in range(1, 21):
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'babi:Task10k:{}'.format(i)
                teacher = Task10kTeacher(opt)
                reply = teacher.act()
                check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_booktest(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.booktest.agents import EvalTeacher, StreamTeacher

        opt = ParlaiParser().parse_args(args=self.args)

        opt['datatype'] = 'train:ordered'
        teacher = StreamTeacher(opt)
        reply = teacher.act()
        check(opt, reply)

        for dt in ['valid', 'test']:
            opt['datatype'] = dt
            teacher = EvalTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_cbt(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.cbt.agents import (NETeacher, CNTeacher, VTeacher,
                                             PTeacher)

        opt = ParlaiParser().parse_args(args=self.args)
        for teacher_class in (NETeacher, CNTeacher, VTeacher, PTeacher):
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                teacher = teacher_class(opt)
                reply = teacher.act()
                check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_cornell_movie(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.cornell_movie.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_dbll_babi(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.dbll_babi.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)
        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'dbll_babi:task:{}_p{}'.format(i, 0.5)
                teacher = TaskTeacher(opt)
                reply = teacher.act()
                check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_dbll_movie(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.dbll_movie.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)

        from parlai.tasks.dbll_movie.agents import KBTeacher
        teacher = KBTeacher(opt)
        reply = teacher.act()
        check_no_labels(opt, reply)

        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'dbll_movie:task:{}_p{}'.format(i, 0.5)
                teacher = TaskTeacher(opt)
                reply = teacher.act()
                check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_dialog_babi(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.dialog_babi.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)

        from parlai.tasks.dialog_babi.agents import KBTeacher
        teacher = KBTeacher(opt)
        reply = teacher.act()
        check_no_labels(opt, reply)

        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'dialog_babi:task:{}'.format(i)
                teacher = TaskTeacher(opt)
                reply = teacher.act()
                check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_mctest(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.mctest.agents import Task160Teacher, Task500Teacher

        opt = ParlaiParser().parse_args(args=self.args)

        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = Task160Teacher(opt)
            reply = teacher.act()
            check(opt, reply)

            teacher = Task500Teacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_moviedialog(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.moviedialog.agents import TaskTeacher, tasks

        opt = ParlaiParser().parse_args(args=self.args)

        from parlai.tasks.moviedialog.agents import KBTeacher
        teacher = KBTeacher(opt)
        reply = teacher.act()
        check_no_labels(opt, reply)

        for i in tasks.keys():
            for dt in ['train:ordered', 'valid', 'test']:
                opt['datatype'] = dt
                opt['task'] = 'moviedialog:task:{}'.format(i)
                teacher = TaskTeacher(opt)
                reply = teacher.act()
                check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_mturkwikimovies(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.mturkwikimovies.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_opensubtitles(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.opensubtitles.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_qacnn(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.qacnn.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_qadailymail(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.qadailymail.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_simplequestions(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.simplequestions.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_squad(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.squad.agents import (DefaultTeacher,
                                               HandwrittenTeacher)

        opt = ParlaiParser().parse_args(args=self.args)

        for dt in ['train:ordered', 'valid']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

            teacher = HandwrittenTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_triviaqa(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.triviaqa.agents import WebTeacher, WikipediaTeacher

        opt = ParlaiParser().parse_args(args=self.args)

        for teacher_class in (WebTeacher, WikipediaTeacher):
            for dt in ['train:ordered', 'valid']:
                opt['datatype'] = dt

                teacher = teacher_class(opt)
                reply = teacher.act()
                check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_ubuntu(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.ubuntu.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_webquestions(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.webquestions.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_wikimovies(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.wikimovies.agents import DefaultTeacher, KBTeacher

        opt = ParlaiParser().parse_args(args=self.args)

        teacher = KBTeacher(opt)
        reply = teacher.act()
        check_no_labels(opt, reply)

        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_wikiqa(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.wikiqa.agents import DefaultTeacher

        opt = ParlaiParser().parse_args(args=self.args)
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt
            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_coco_datasets(self):
        # one unit test for tasks with coco so images are only downloaded once
        from parlai.core.params import ParlaiParser
        opt = ParlaiParser().parse_args(args=self.args)

        # VisDial
        from parlai.tasks.visdial.agents import DefaultTeacher
        for dt in ['train:ordered', 'valid']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        # VQA_v1
        from parlai.tasks.vqa_v1.agents import McTeacher, OeTeacher
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = McTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

            teacher = OeTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        # VQA_v2
        from parlai.tasks.vqa_v2.agents import OeTeacher
        for dt in ['train:ordered', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = OeTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_fvqa(self):
        from parlai.core.params import ParlaiParser
        parser = ParlaiParser()
        parser.add_task_args(['-t', 'fvqa'])
        opt = parser.parse_args(args=self.args)

        from parlai.tasks.fvqa.agents import DefaultTeacher
        for dt in ['train:ordered', 'test']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_insuranceqa(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.insuranceqa.agents import V1Teacher, V2Teacher

        opt = ParlaiParser().parse_args(args=self.args)

        for dt in ['train', 'valid', 'test']:
            opt['datatype'] = dt

            teacher = V1Teacher(opt)
            reply = teacher.act()
            check(opt, reply)

            teacher = V2Teacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)

    def test_ms_marco(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.ms_marco.agents import DefaultTeacher, PassageTeacher

        opt = ParlaiParser().parse_args(args=self.args)

        for dt in ['train', 'valid']:
            opt['datatype'] = dt

            teacher = DefaultTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

            teacher = PassageTeacher(opt)
            reply = teacher.act()
            check(opt, reply)

        shutil.rmtree(self.TMP_PATH)


if __name__ == '__main__':
    # clean out temp dir first
    shutil.rmtree(TestData.TMP_PATH, ignore_errors=True)

    tp = unittest.main(exit=False)
    error_code = len(tp.result.errors)
    if error_code != 0:
        print('At least one test failed. Leaving directory ' +
              '{} with temporary files in place '.format(TestData.TMP_PATH) +
              'for inspection (only failed tasks or images remain).')
    else:
        shutil.rmtree(TestData.TMP_PATH, ignore_errors=True)
    sys.exit(error_code)
