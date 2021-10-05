#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class StaticKGTeacher(AutoTeacherTest):
    task = 'jericho_world'


class TestStaticKGTeacher(AutoTeacherTest):
    task = 'jericho_world:StaticKGTeacher'


class TestActionKGTeacher(AutoTeacherTest):
    task = 'jericho_world:ActionKGTeacher'


class TestStateToValidActionsTeacher(AutoTeacherTest):
    task = 'jericho_world:StateToValidActionsTeacher'


class TestStateToActionTeacher(AutoTeacherTest):
    task = 'jericho_world:StateToActionTeacher'
