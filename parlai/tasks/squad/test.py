#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "squad"


class TestIndexTeacher(AutoTeacherTest):
    task = "squad:index"


class TestOpensquadTeacher(AutoTeacherTest):
    task = "squad:opensquad"


class TestFulldocTeacher(AutoTeacherTest):
    task = "squad:fulldoc"


class TestSentenceTeacher(AutoTeacherTest):
    task = "squad:sentence"


class TestFulldocsentenceTeacher(AutoTeacherTest):
    task = "squad:fulldocsentence"
