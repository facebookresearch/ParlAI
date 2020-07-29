#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest


class TestDefaultTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2'


class TestNormalizedTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:normalized'


class TestBothTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:both'


class TestNoneTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:none'


class TestSelfRevisedTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:self_revised'
