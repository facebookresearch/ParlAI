#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .build import build
from parlai.tasks.dream.agents import BaseMultipleChoiceTeacher
import os


class C3Teacher(BaseMultipleChoiceTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        jsons_path = os.path.join(opt['datapath'], 'C3')
        self.id = 'c3'
        super().__init__(opt, jsons_path, shared)


class DefaultTeacher(C3Teacher):
    pass
