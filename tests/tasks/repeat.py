#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Class which creates a dummy dataset for testing purposes.
   Used in test_train_model.py
"""
from parlai.core.teachers import DialogTeacher

import copy


class RepeatTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = 'unused_path'
        task = opt.get('task', 'tests.tasks.repeat:RepeatTeacher:50')
        self.data_length = int(task.split(':')[2])
        super().__init__(opt, shared)

    def setup_data(self, unused_path):
        for i in range(self.data_length):
            yield ((str(i), [str(i)]), True)

    def num_examples(self):
        return self.data_length

    def num_episodes(self):
        return self.data_length
