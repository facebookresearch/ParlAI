#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class TestDefaultTeacher(AutoTeacherTest):
    task = 'cornell_movie'


class TestAll1kTeacher(AutoTeacherTest):
    task = 'cornell_movie:double'
