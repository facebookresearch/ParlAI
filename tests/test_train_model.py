# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from examples.train_model import TrainLoop, setup_args
from parlai.core.agents import create_agent
from parlai.core.utils import Timer
from parlai.core.worlds import create_task

import ast
import importlib
import unittest
import sys


class TestTrainModel(unittest.TestCase):
    """Basic tests on the train_model.py example."""

    def setup_test_args(self):
        parser = setup_args(model_args=['--model', 'memnn'])
        # using memnn, so we want to check if torch is downloaded
        torch_downloaded = importlib.find_loader('torch')
        self.assertTrue(torch_downloaded is not None, "Torch not downloaded")
        return parser

    def test_output(self):
        class TestTrainLoop(TrainLoop):
            args = [
                '--model', 'memnn',
                '--task', 'tests.tasks.repeat:RepeatTeacher:10',
                '--dict-file', '/tmp/repeat',
                '-bs', '1',
                '-vtim', '5',
                '-vp', '2',
                '--embedding-size', '8',
                '--no-cuda'
            ]

            def __init__(self, parser):
                opt = parser.parse_args(self.args, print_args=False)
                self.agent = create_agent(opt)
                self.world = create_task(opt, self.agent)
                self.train_time = Timer()
                self.validate_time = Timer()
                self.log_time = Timer()
                self.save_time = Timer()
                print('[ training... ]')
                self.parleys = 0
                self.max_num_epochs = opt['num_epochs'] \
                    if opt['num_epochs'] > 0 else float('inf')
                self.max_train_time = opt['max_train_time'] \
                    if opt['max_train_time'] > 0 else float('inf')
                self.log_every_n_secs = opt['log_every_n_secs'] \
                    if opt['log_every_n_secs'] > 0 else float('inf')
                self.val_every_n_secs = opt['validation_every_n_secs'] \
                    if opt['validation_every_n_secs'] > 0 else float('inf')
                self.save_every_n_secs = opt['save_every_n_secs'] \
                    if opt['save_every_n_secs'] > 0 else float('inf')
                self.best_valid = 0
                self.impatience = 0
                self.saved = False
                self.valid_world = None
                self.opt = opt

        class display_output(object):
            def __init__(self):
                self.data = []

            def write(self, s):
                self.data.append(s)

            def __str__(self):
                return "".join(self.data)

        old_out = sys.stdout
        output = display_output()
        try:
            sys.stdout = output
            TestTrainLoop(self.setup_test_args()).train()
        finally:
            # restore sys.stdout
            sys.stdout = old_out

        str_output = str(output)

        self.assertTrue(len(str_output) > 0, "Output is empty")
        self.assertTrue("[ training... ]" in str_output,
                        "Did not reach training step")
        self.assertTrue("[ running eval: valid ]" in str_output,
                        "Did not reach validation step")
        self.assertTrue("valid:{'total': 10," in str_output,
                        "Did not complete validation")
        self.assertTrue("[ running eval: test ]" in str_output,
                        "Did not reach evaluation step")
        self.assertTrue("test:{'total': 10," in str_output,
                        "Did not complete evaluation")

        list_output = str_output.split("\n")
        for line in list_output:
            if "test:{" in line:
                score = ast.literal_eval(line.split("test:", 1)[1])
                self.assertTrue(score['accuracy'] == 1,
                                "Accuracy did not reach 1")

if __name__ == '__main__':
    unittest.main()
