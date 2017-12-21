# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
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
