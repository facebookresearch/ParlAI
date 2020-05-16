#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .build import build
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.dream.agents import setup_data, num_examples, get
import os


class C3Teacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        build(opt)
        jsons_path = os.path.join(opt['datapath'], 'C3')
        self.id = 'c3'
        self.episodes = setup_data(opt, jsons_path)
        self.reset()

    def num_examples(self):
        return num_examples(self.episodes)

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=0):
        return get(self.id, self.episodes, entry_idx)


class DefaultTeacher(C3Teacher):
    pass
