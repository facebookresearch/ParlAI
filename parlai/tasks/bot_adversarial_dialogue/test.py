#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest


class TestBotAdversarialDialogueTurn4Teacher(unittest.TestCase, AutoTeacherTest):
    task = 'bot_adversarial_dialogue:bad_speaker_to_eval=all:bad_safety_mix=all:bad_num_turns=4'


class TestBotAdversarialDialogueSafeTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'bot_adversarial_dialogue:bad_speaker_to_eval=all:bad_safety_mix=safe:bad_num_turns=4'


class TestBotAdversarialDialogueHumanTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'bot_adversarial_dialogue:bad_speaker_to_eval=human:bad_safety_mix=all:bad_num_turns=4'


class TestHumanSafetyEvaluation(unittest.TestCase, AutoTeacherTest):
    task = 'bot_adversarial_dialogue:HumanSafetyEvaluation'
