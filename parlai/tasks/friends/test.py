#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class TestDefaultTeacher(AutoTeacherTest):
    task = 'friends'


class TestAllCharactersTeacher(AutoTeacherTest):
    task = 'friends:all_characters'


class TestRachelTeacher(AutoTeacherTest):
    task = 'friends:rachel'


class TestMonicaTeacher(AutoTeacherTest):
    task = 'friends:monica'


class TestPhoebeTeacher(AutoTeacherTest):
    task = 'friends:phoebe'


class TestJoeyTeacher(AutoTeacherTest):
    task = 'friends:joey'


class TestChandlerTeacher(AutoTeacherTest):
    task = 'friends:chandler'


class TestRossTeacher(AutoTeacherTest):
    task = 'friends:ross'
