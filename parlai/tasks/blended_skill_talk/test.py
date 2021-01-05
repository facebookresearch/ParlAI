#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "blended_skill_talk"


class TestBlendedSkillTalkTeacher(AutoTeacherTest):
    task = "blended_skill_talk:blended_skill_talk"


class TestConvAI2PersonaTopicifierTeacher(AutoTeacherTest):
    task = "blended_skill_talk:convAI2_persona_topicifier"


class TestWoWPersonaTopicifierTeacher(AutoTeacherTest):
    task = "blended_skill_talk:wo_w_persona_topicifier"


class TestEDPersonaTopicifierTeacher(AutoTeacherTest):
    task = "blended_skill_talk:e_d_persona_topicifier"


class TestAllTeacher(AutoTeacherTest):
    task = "blended_skill_talk:all"
