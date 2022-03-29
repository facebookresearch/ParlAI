#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class NQAutoTeacherTest(AutoTeacherTest):
    def test_train_stream_ordered(self, data_regression):
        """
        Ignore train set because it's so large.
        """
        pass


class TestNaturalQuestionsTeacher(NQAutoTeacherTest):
    task = 'natural_questions'


class TestNaturalQuestionsSampleTeacher(AutoTeacherTest):
    task = 'natural_questions:NaturalQuestionsSampleTeacher'


class TestNaturalQuestionsOpenTeacher(AutoTeacherTest):
    task = 'natural_questions:NaturalQuestionsOpenTeacher'
