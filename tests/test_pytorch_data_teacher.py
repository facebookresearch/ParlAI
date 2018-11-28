#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.scripts.display_data import setup_args
from parlai.core.agents import create_task_agent_from_taskname

import unittest
import io
from torch.utils.data.sampler import RandomSampler, SequentialSampler as Sequential
from contextlib import redirect_stdout


class TestPytorchDataTeacher(unittest.TestCase):
    """Various tests for PytorchDataTeacher"""

    def test_shuffle(self):
        """Simple test to ensure that dataloader is initialized with correct
            data sampler
        """
        dts = ['train', 'valid', 'test']
        exts = ['', ':stream', ':ordered', ':stream:ordered']
        shuffle_opts = [False, True]
        task = 'babi:task1k:1'
        for dt in dts:
            for ext in exts:
                datatype = dt + ext
                for shuffle in shuffle_opts:
                    opt_defaults = {
                        'pytorch_teacher_task': task,
                        'datatype': datatype,
                        'shuffle': shuffle
                    }
                    print('Testing {} with args {}'.format(task,
                                                           opt_defaults))
                    f = io.StringIO()
                    with redirect_stdout(f):
                        parser = setup_args()
                        parser.set_defaults(**opt_defaults)
                        opt = parser.parse_args()
                        teacher = create_task_agent_from_taskname(opt)[0]
                    if ('ordered' in datatype or
                            ('stream' in datatype and not opt.get('shuffle')) or
                            'train' not in datatype):
                        self.assertTrue(
                            type(teacher.pytorch_dataloader.sampler) is Sequential,
                            'PytorchDataTeacher failed with args: {}'.format(opt)
                        )
                    else:
                        self.assertTrue(
                            type(teacher.pytorch_dataloader.sampler) is RandomSampler,
                            'PytorchDataTeacher failed with args: {}'.format(opt)
                        )


if __name__ == '__main__':
    unittest.main()
