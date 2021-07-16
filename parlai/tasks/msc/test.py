#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class TestMscTeacher(AutoTeacherTest):
    task = 'msc'


class TestMscAllTeacher(AutoTeacherTest):
    task = 'msc:include_last_session=True'


class TestPersonaSummaryTeacher(AutoTeacherTest):
    task = 'msc:PersonaSummaryTeacher'


class TestPersonaSummaryAllTeacher(AutoTeacherTest):
    task = 'msc:PersonaSummaryTeacher:include_last_session=True'
