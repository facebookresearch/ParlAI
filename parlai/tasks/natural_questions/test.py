#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class NQAutoTeacherTest(AutoTeacherTest):
    def test_train_stream_ordered(self, data_regression):
        """
        Test --datatype train:stream:ordered.
        """
        pass


class TestNaturalQuestionsTeacher(NQAutoTeacherTest):
    task = 'natural_questions'  # replace with your teacher name


class TestNaturalQuestionsSampleTeacher(AutoTeacherTest):
    task = (
        'natural_questions:NaturalQuestionsSampleTeacher'
    )  # replace with your teacher name


class TestNaturalQuestionsOpenTeacher(AutoTeacherTest):
    task = (
        'natural_questions:NaturalQuestionsOpenTeacher'
    )  # replace with your teacher name
