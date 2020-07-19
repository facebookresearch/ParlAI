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

    def test_with_stream(self):
        """
        Test that multi-tasking works with datatype train:stream.
        """
        task1 = 'integration_tests:infinite_train'
        task2 = 'integration_tests:short_fixed'
        opt = ParlaiParser(True, True).parse_kwargs(
            task=f'{task1},{task2}',
            model='fixed_response',
            fixed_response='Hello!',
            datatype='train:stream',
            batchsize=16,
        )
        agent = create_agent(opt)
        world = create_task(opt, agent)

        for i in range(100):
            world.parley()
            if i % 10 == 0 and i > 0:
                teacher_acts, _ = world.get_acts()
                for act in teacher_acts:
                    act_id = act.get('id')
                    assert 'text' in act, f'Task {act_id} acts are empty'
                report = world.report()
                for task in [task1, task2]:
                    err = f'Task {task} has no examples on iteration {i}'
                    assert f'{task}/exs' in report, err
                    exs = report[f'{task}/exs'].value()
                    assert exs > 0, err
                world.reset_metrics()
