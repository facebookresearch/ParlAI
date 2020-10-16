#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest


class TestBlendedSkillTalkTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_blended_skill_talk'


class TestConvAI2Teacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_convAI2_persona_topicifier'


class TestEDTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_ED_persona_topicifier'


class TestWoWTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_WoW_persona_topicifier'
