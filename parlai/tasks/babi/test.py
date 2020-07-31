#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest


class TestDefaultTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'babi'


class TestAll1kTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'babi:all1k'


class TestAll10kTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'babi:all10k'
