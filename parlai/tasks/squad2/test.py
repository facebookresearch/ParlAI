#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "squad2"


class TestIndexTeacher(AutoTeacherTest):
    task = "squad2:index"


class TestOpenSquadTeacher(AutoTeacherTest):
    task = "squad2:open_squad"


class TestSentenceIndexTeacher(AutoTeacherTest):
    task = "squad2:sentence_index"


class TestSentenceIndexEditTeacher(AutoTeacherTest):
    task = "squad2:sentence_index_edit"


class TestSentenceLabelsTeacher(AutoTeacherTest):
    task = "squad2:sentence_labels"
