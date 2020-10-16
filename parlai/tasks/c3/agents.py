#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .build import build
from parlai.tasks.dream.agents import DREAMTeacher
import os


class C3Teacher(DREAMTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'c3'

    def setup_data(self):
        build(self.opt)
        jsons_path = os.path.join(self.opt['datapath'], 'C3')
        return self.setup_helper(jsons_path)


class DefaultTeacher(C3Teacher):
    pass
