#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class TestBotAdversarialDialogueTurn4Teacher(AutoTeacherTest):
    task = 'bot_adversarial_dialogue:bad_speaker_to_eval=all:bad_safety_mix=all:bad_num_turns=4'


class TestBotAdversarialDialogueSafeTeacher(AutoTeacherTest):
    task = 'bot_adversarial_dialogue:bad_speaker_to_eval=all:bad_safety_mix=safe:bad_num_turns=4'


class TestBotAdversarialDialogueHumanTeacher(AutoTeacherTest):
    task = 'bot_adversarial_dialogue:bad_speaker_to_eval=human:bad_safety_mix=all:bad_num_turns=4'


class TestHumanSafetyEvaluation(AutoTeacherTest):
    task = 'bot_adversarial_dialogue:HumanSafetyEvaluation'
