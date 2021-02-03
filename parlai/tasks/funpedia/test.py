#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "funpedia"


class TestFunpediaTeacher(AutoTeacherTest):
    task = "funpedia:funpedia"


class TestNopersonaTeacher(AutoTeacherTest):
    task = "funpedia:nopersona"


class TestLmTeacher(AutoTeacherTest):
    task = "funpedia:lm"


class TestEchoTeacher(AutoTeacherTest):
    task = "funpedia:echo"


class TestSentencechooseTeacher(AutoTeacherTest):
    task = "funpedia:sentencechoose"
