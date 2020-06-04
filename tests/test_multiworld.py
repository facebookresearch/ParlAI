#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.core.teachers import DialogTeacher, register_teacher
import parlai.utils.testing as testing_utils


@register_teacher('teacher1')
class Teacher1(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = 'filler'
        self.id = 'teacher1'
        super().__init__(opt, shared)

    def setup_data(self, datafile):
        for i in range(100):
            yield (str(i), str(i)), False
            yield (str(i), str(i)), True


@register_teacher('teacher2')
class Teacher2(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = 'filler'
        self.id = 'teacher2'
        super().__init__(opt, shared)

    def setup_data(self, datafile):
        for i in range(10):
            yield (str(i), str(i)), True


class TestMultiworld(unittest.TestCase):
    @testing_utils.retry(ntries=10)
    def test_equal(self):
        opt = ParlaiParser(True, True).parse_kwargs(
            task='teacher1,teacher2',
            multitask_weights='1,1',
            model='fixed_response',
            fixed_response='None',
            datatype='train',
            batchsize=1,
        )
        agent = create_agent(opt)
        world = create_task(opt, agent)

        for _ in range(1000):
            world.parley()

        report = world.report()
        ratio = report['teacher1/exs'].value() / report['teacher2/exs'].value()
        assert ratio > 1.7
        assert ratio < 2.3

    @testing_utils.retry(ntries=10)
    def test_stochastic(self):
        opt = ParlaiParser(True, True).parse_kwargs(
            task='teacher1,teacher2',
            multitask_weights='stochastic',
            model='fixed_response',
            fixed_response='None',
            datatype='train',
            batchsize=1,
        )
        agent = create_agent(opt)
        world = create_task(opt, agent)

        for _ in range(1000):
            world.parley()

        report = world.report()
        # stochastic so wide range
        ratio = report['teacher1/exs'].value() / report['teacher2/exs'].value()
        assert ratio > 18
        assert ratio < 22
