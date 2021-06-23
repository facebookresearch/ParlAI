#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from parlai.core.loader import load_teacher_module
from parlai.core.opt import Opt
from parlai.tasks.wizard_of_wikipedia.rare_f1 import RareF1Computer


class TestRareF1(unittest.TestCase):
    def test_basic_init(self):
        rare_f1 = RareF1Computer(
            'he was a man no he was a dragon man or maybe he was just a dragon'
        )
        score = rare_f1.compute('trogdor', answers=['he was trogdor'])
        self.assertEqual(score.value(), 1.0)

    def test_from_dialog_data(self):
        test_teacher_class = load_teacher_module(taskname='integration_tests')
        test_teacher = test_teacher_class(opt=Opt(datatype='train:ordered'))
        rare_f1 = RareF1Computer.from_reference_dialog_data(test_teacher.data)
        score = rare_f1.compute('textpectation', answers=['textpectation'])
        self.assertEqual(score.value(), 1.0)

    def test_from_teacher(self):
        test_teacher_class = load_teacher_module(taskname='integration_tests')
        test_teacher = test_teacher_class(opt=Opt(datatype='train:ordered'))
        rare_f1 = RareF1Computer.from_reference_teacher(test_teacher)
        score = rare_f1.compute('textpectation', answers=['textpectation'])
        self.assertEqual(score.value(), 1.0)
