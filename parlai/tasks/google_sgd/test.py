#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = "google_sgd"


class TestText2API2TextTeacher(AutoTeacherTest):
    task = "google_sgd:text2_a_p_i2_text"


class TestText2TextTeacher(AutoTeacherTest):
    task = "google_sgd:text2_text"
