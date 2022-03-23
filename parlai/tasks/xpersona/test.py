#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class TestDefaultTeacher(AutoTeacherTest):
    task = 'xpersona'


class TestEnTeacher(AutoTeacherTest):
    task = 'xpersona:En'


class TestZhTeacher(AutoTeacherTest):
    task = 'xpersona:Zh'


class TestFrTeacher(AutoTeacherTest):
    task = 'xpersona:Fr'


class TestIdTeacher(AutoTeacherTest):
    task = 'xpersona:Id'


class TestItTeacher(AutoTeacherTest):
    task = 'xpersona:It'


class TestKoTeacher(AutoTeacherTest):
    task = 'xpersona:Ko'


class TestJpTeacher(AutoTeacherTest):
    task = 'xpersona:Jp'
