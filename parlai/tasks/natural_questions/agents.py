#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

class LongAnswerTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        self.id = 'natural_questions'
        super().__init__(opt, shared)

    def setup_data(self, path):
        raise NotImplementedError

class DefaultTeacher(LongAnswerTeacher):
    pass
