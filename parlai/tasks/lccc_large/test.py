#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the

from parlai.utils.testing import AutoTeacherTest  # noqa: F401


class TestDefaultTeacher(AutoTeacherTest):
    task = 'lccc_large'
