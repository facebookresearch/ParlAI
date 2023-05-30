#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = 'light_multiparty'


class TestSpeakerPredictionTeacher(AutoTeacherTest):
    task = f'light_multiparty:SpeakerPredictionTeacher'


class TestSpeakersTeacher(AutoTeacherTest):
    task = f'light_multiparty:SpeakerPredictionTeacher'


class TestFirstSpeakerTeacher(AutoTeacherTest):
    task = f'light_multiparty:FirstSpeakerTeacher'


class TestSecondSpeakerTeacher(AutoTeacherTest):
    task = f'light_multiparty:SecondSpeakerTeacher'


class TestThirdSpeakerTeacher(AutoTeacherTest):
    task = f'light_multiparty:ThirdSpeakerTeacher'
