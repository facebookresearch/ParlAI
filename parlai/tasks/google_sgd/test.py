#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class DisabledTestDefaultTeacher(AutoTeacherTest):
    task = "google_sgd"


class DisabledTestText2API2TextTeacher(AutoTeacherTest):
    task = "google_sgd:text2_a_p_i2_text"


class DisabledTestText2TextTeacher(AutoTeacherTest):
    task = "google_sgd:text2_text"
