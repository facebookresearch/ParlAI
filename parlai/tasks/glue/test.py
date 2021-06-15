#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "glue"


class TestColaTeacher(AutoTeacherTest):
    task = "glue:cola"


class TestMnliTeacher(AutoTeacherTest):
    task = "glue:mnli"


class TestMrpcTeacher(AutoTeacherTest):
    task = "glue:mrpc"


class TestQnliTeacher(AutoTeacherTest):
    task = "glue:qnli"


class TestQqpTeacher(AutoTeacherTest):
    task = "glue:qqp"


class TestRteTeacher(AutoTeacherTest):
    task = "glue:rte"


class TestSst2Teacher(AutoTeacherTest):
    task = "glue:sst2"


class TestStsbTeacher(AutoTeacherTest):
    task = "glue:stsb"


class TestWnliTeacher(AutoTeacherTest):
    task = "glue:wnli"
