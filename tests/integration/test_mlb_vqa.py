#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.scripts.train_model import TrainLoop, setup_args

import unittest
import io
from contextlib import redirect_stdout


class TestTrainModel(unittest.TestCase):
    """Basic tests on the train_model.py example."""

    def test_output(self):
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                import torch  # noqa: F401
            except ImportError:
                print('Cannot import torch, skipping test_train_model')
                return
            parser = setup_args()
            parser.set_defaults(
                model='mlb_vqa',
                pytorch_teacher_dataset='vqa_v1',
                image_mode='resnet152_spatial',
                image_size=448,
                image_cropsize=448,
                dict_file='/tmp/vqa_v1',
                batchsize=1,
                num_epochs=1,
                no_cuda=True,
                use_hdf5=False,
                pytorch_preprocess=False,
                batch_sort_cache='none',
                numworkers=1,
                unittest=True
            )
            TrainLoop(parser).train()

        str_output = f.getvalue()
        self.assertTrue(len(str_output) > 0, "Output is empty")
        self.assertTrue("[ training... ]" in str_output,
                        "Did not reach training step")
        self.assertTrue("[ running eval: valid ]" in str_output,
                        "Did not reach validation step")
        self.assertTrue("valid:{'exs': 10," in str_output,
                        "Did not complete validation")
        self.assertTrue("[ running eval: test ]" in str_output,
                        "Did not reach evaluation step")
        self.assertTrue("test:{'exs': 0}" in str_output,
                        "Did not complete evaluation")


if __name__ == '__main__':
    unittest.main()
