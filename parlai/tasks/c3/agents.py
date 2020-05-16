#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .build import build
from parlai.tasks.dream.agents import BaseMultipleChoiceTeacher
import os


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    jsons_path = os.path.join(opt['datapath'], 'C3')
    return jsons_path


class C3Teacher(BaseMultipleChoiceTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, _path, shared)
        self.id = 'c3'


class DefaultTeacher(C3Teacher):
    pass
