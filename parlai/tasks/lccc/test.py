#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest, skipIfCircleCI


class TestDefaultTeacher(AutoTeacherTest):
    task = 'lccc'

    @skipIfCircleCI
    def test_train_stream_ordered(self, data_regression):
        return super().test_train_stream_ordered(data_regression)
