#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the stack that keeps track of model image chats.
"""

import random

import numpy as np
import torch
from pytest_regressions.file_regression import FileRegressionFixture

try:

    from parlai.crowdsourcing.tasks.model_chat.utils import ImageStack
    import parlai.utils.testing as testing_utils

    class TestImageStack:
        """
        Test the stack that keeps track of model image chats.
        """

        def test_fill_stack(self, file_regression: FileRegressionFixture):
            """
            Check the expected output when filling up the stack.

            Request image/model slots from the stack, and check that the behavior is as
            expected.
            """

            seed = 0
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            with testing_utils.tempdir() as tmpdir:

                # Params
                opt = {
                    'evals_per_image_model_combo': 2,
                    'models': ['model_1', 'model_2'],
                    'num_images': 3,
                    'stack_folder': tmpdir,
                }
                num_stack_slots = (
                    opt['evals_per_image_model_combo']
                    * len(opt['models'])
                    * opt['num_images']
                )
                num_workers = 5
                worker_id_to_remove = '2'
                stack_idx_to_remove_worker_from = 0

                # Create the stack
                stack = ImageStack(opt)

                with testing_utils.capture_output() as output:
                    for _ in range(num_stack_slots):
                        worker_id = random.randrange(num_workers)
                        _ = stack.get_next_image(str(worker_id))
                        print('STACK: ', stack.stack)
                    stack.remove_worker_from_stack(
                        worker=worker_id_to_remove,
                        stack_idx=stack_idx_to_remove_worker_from,
                    )
                    print('STACK: ', stack.stack)
                    stdout = output.getvalue()

                # Check the output against what it should be
                file_regression.check(contents=stdout)

except ImportError:
    pass
