#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class TestNaturalQuestionsTeacher(AutoTeacherTest):
    task = 'natural_questions'  # replace with your teacher name


class TestNaturalQuestionsOpenTeacher(AutoTeacherTest):
    task = (
        'natural_questions:NaturalQuestionsOpenTeacher'
    )  # replace with your teacher name
