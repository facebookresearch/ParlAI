#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "personachat"


class TestNoneTeacher(AutoTeacherTest):
    task = "personachat:none"


class TestSelfOriginalTeacher(AutoTeacherTest):
    task = "personachat:self_original"


class TestSelfTeacher(AutoTeacherTest):
    task = "personachat:self"


class TestSelfRevisedTeacher(AutoTeacherTest):
    task = "personachat:self_revised"


class TestOtherOriginalTeacher(AutoTeacherTest):
    task = "personachat:other_original"


class TestOtherTeacher(AutoTeacherTest):
    task = "personachat:other"


class TestOtherRevisedTeacher(AutoTeacherTest):
    task = "personachat:other_revised"


class TestBothOriginalTeacher(AutoTeacherTest):
    task = "personachat:both_original"


class TestBothTeacher(AutoTeacherTest):
    task = "personachat:both"


class TestBothRevisedTeacher(AutoTeacherTest):
    task = "personachat:both_revised"
