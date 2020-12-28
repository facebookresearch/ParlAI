#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the stack that keeps track of model image chats.
"""

import os
import random
import unittest

import numpy as np
import pandas as pd
import torch

from parlai.crowdsourcing.tasks.model_chat.utils import ImageStack
import parlai.utils.testing as testing_utils


class TestImageStack(unittest.TestCase):
    """
    Test the stack that keeps track of model image chats.
    """

    def test_load_stack(self):
        """
        Check the expected output when loading the stack.

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
            stack_idx_to_remove_worker_from = 0  # TODO: set

            # Create the stack
            stack = ImageStack(opt)

            with testing_utils.capture_output() as output:
                for _ in range(num_stack_slots):
                    worker_id = random.randrange(num_workers)
                    _ = stack.get_next_image(str(worker_id))
                stack.remove_worker_from_stack(
                    worker=worker_id_to_remove,
                    stack_idx=stack_idx_to_remove_worker_from,
                )
                actual_stdout = output.getvalue()

            # Check the output against what it should be
            desired_stdout = f"""\
Oh but obviously I am wrong\
"""  # TODO: fix
            actual_stdout_lines = actual_stdout.split('\n')
            for desired_line in desired_stdout.split('\n'):
                if desired_line not in actual_stdout_lines:
                    raise ValueError(
                        f'\n\tThe following line:\n\n{desired_line}\n\n\twas not found '
                        f'in the actual stdout:\n\n{actual_stdout}'
                    )


if __name__ == "__main__":
    unittest.main()
