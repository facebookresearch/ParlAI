#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "taskmaster"


class TestSelfDialogueTeacher(AutoTeacherTest):
    task = "taskmaster:self_dialogue"


class TestWozDialogueTeacher(AutoTeacherTest):
    task = "taskmaster:woz_dialogue"


class TestSelfDialogueSegmentTeacher(AutoTeacherTest):
    task = "taskmaster:self_dialogue_segment"


class TestSystemTeacher(AutoTeacherTest):
    task = "taskmaster:SystemTeacher"


class TestUserSimulatorTeacher(AutoTeacherTest):
    task = "taskmaster:UserSimulatorTeacher"
