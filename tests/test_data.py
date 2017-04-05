#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import unittest

class TestData(unittest.TestCase):
    """Test access to different datasets."""

    def test_babi_1k(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.babi.teachers import Task1kTeacher

        opt = ParlaiParser().parse_args(args=[
            '--datatype', 'train:ordered'
        ])
        for i in range(1, 21):
            opt['task'] = 'babi:Task1k:{}'.format(i)
            teacher = Task1kTeacher(opt)
            reply = teacher.act({})
            assert reply
            assert 'text' in reply
            assert 'labels' in reply
            assert 'done' in reply
            assert len(reply['labels']) > 0

    def test_babi_10k(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.babi.teachers import Task10kTeacher

        opt = ParlaiParser().parse_args(args=[
            '--datatype', 'train:ordered'
        ])
        for i in range(1, 21):
            opt['task'] = 'babi:Task10k:{}'.format(i)
            teacher = Task10kTeacher(opt)
            reply = teacher.act({})
            assert reply
            assert 'text' in reply
            assert 'labels' in reply
            assert 'done' in reply
            assert len(reply['labels']) > 0

    def test_squad(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.squad.teachers import DefaultTeacher

        opt = ParlaiParser().parse_args(args=[
            '--datatype', 'train:ordered'
        ])
        opt['task'] = 'squad'
        teacher = DefaultTeacher(opt)
        reply = teacher.act({})
        assert reply
        assert 'text' in reply
        assert 'labels' in reply
        assert 'done' in reply
        assert len(reply['labels']) > 0

    def test_squad_inherit(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.squad.teachers import InheritedSquadTeacher

        opt = ParlaiParser().parse_args(args=[
            '--datatype', 'train:ordered'
        ])
        opt['task'] = 'squad'
        teacher = InheritedSquadTeacher(opt)
        reply = teacher.act({})
        assert reply
        assert 'text' in reply
        assert 'labels' in reply
        assert 'done' in reply
        assert len(reply['labels']) > 0

    def test_wikiqa(self):
        from parlai.core.params import ParlaiParser
        from parlai.tasks.wikiqa.teachers import DefaultTeacher

        opt = ParlaiParser().parse_args(args=[
            '--datatype', 'train:ordered'
        ])
        opt['task'] = 'squad'
        teacher = DefaultTeacher(opt)
        reply = teacher.act({})
        assert reply
        assert 'text' in reply
        assert 'labels' in reply
        assert 'done' in reply
        assert len(reply['labels']) > 0


if __name__ == '__main__':
    unittest.main()
