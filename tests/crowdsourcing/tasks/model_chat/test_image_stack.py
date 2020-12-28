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

            # Create the stack
            stack = ImageStack(opt)

            # TODO: revise below

            with testing_utils.capture_output() as output:
                compiler = TurnAnnotationsStaticResultsCompiler(opt)
                compiler.NUM_SUBTASKS = 3
                compiler.NUM_ANNOTATIONS = 3
                compiler.compile_results()
                actual_stdout = output.getvalue()

            # Check the output against what it should be
            desired_stdout = f"""\
Got 4 folders to read.
Average task completion time (seconds) was: 187.5
Returning master dataframe with 48 annotations.
Dropped 1 inconsistently annotated utterances (none_all_good and a problem bucket). Now have 7 utterances.
Removed 1 that did not have annotations by 3 workers. 6 annotations remaining.
summed_df has length 2; bot_only_df: 4
Number of unique utterance_ids: 2.
Bucket: bucket_0, total unique problem utterances: 1 (50.0% of all), one annotator: 1 (100.0%), two_annotators: 0 (0.0%), three+ annotators: 0 (0.0%)
Bucket: bucket_4, total unique problem utterances: 1 (50.0% of all), one annotator: 0 (0.0%), two_annotators: 1 (100.0%), three+ annotators: 0 (0.0%)
Bucket: none_all_good, total unique problem utterances: 1 (50.0% of all), one annotator: 1 (100.0%), two_annotators: 0 (0.0%), three+ annotators: 0 (0.0%)
Bucket: any_problem, total unique problem utterances: 2 (100.0% of all), one annotator: 1 (50.0%), two_annotators: 1 (50.0%), three+ annotators: 0 (0.0%)
Got 4 utterances with gold annotations. Found 8 utterances matching gold annotations from DataFrame.
Average agreement with 4 total gold utterances annotated was:
bucket_0: 91.7% (0 gold problem samples)
bucket_1: 100.0% (1 gold problem samples)
bucket_2: 75.0% (0 gold problem samples)
bucket_3: 91.7% (0 gold problem samples)
bucket_4: 100.0% (2 gold problem samples)
none_all_good: 58.3% (1 gold problem samples)
Average agreement problem samples only with 4 total gold utterances annotated was:
bucket_0: nan% (0 gold problem samples)
bucket_1: 100.0% (1 gold problem samples)
bucket_2: nan% (0 gold problem samples)
bucket_3: nan% (0 gold problem samples)
bucket_4: 100.0% (2 gold problem samples)
none_all_good: 33.3% (1 gold problem samples)
Calculating agreement on 8 annotations.
Fleiss' kappa for bucket_0 is: -0.410.
Fleiss' kappa for bucket_1 is: -0.385.
Fleiss' kappa for bucket_2 is: -0.385.
Fleiss' kappa for bucket_3 is: -0.410.
Fleiss' kappa for bucket_4 is: -0.380.
Fleiss' kappa for none_all_good is: -0.410.\
"""
            actual_stdout_lines = actual_stdout.split('\n')
            for desired_line in desired_stdout.split('\n'):
                if desired_line not in actual_stdout_lines:
                    raise ValueError(
                        f'\n\tThe following line:\n\n{desired_line}\n\n\twas not found '
                        f'in the actual stdout:\n\n{actual_stdout}'
                    )

            # Check that the saved results file is what it should be
            sort_columns = ['hit_id', 'worker_id', 'conversation_id', 'turn_idx']
            desired_results_path = os.path.join(
                analysis_config_folder, 'desired_results.csv'
            )
            desired_results = (
                pd.read_csv(desired_results_path)
                .drop('folder', axis=1)
                .sort_values(sort_columns)
                .reset_index(drop=True)
            )
            # Drop the 'folder' column, which contains a system-dependent path string
            actual_results_rel_path = [
                obj for obj in os.listdir(tmpdir) if obj.startswith('results')
            ][0]
            actual_results_path = os.path.join(tmpdir, actual_results_rel_path)
            actual_results = (
                pd.read_csv(actual_results_path)
                .drop('folder', axis=1)
                .sort_values(sort_columns)
                .reset_index(drop=True)
            )
            if not actual_results.equals(desired_results):
                raise ValueError(
                    f'\n\n\tDesired results:\n{desired_results.to_csv()}'
                    f'\n\n\tActual results:\n{actual_results.to_csv()}'
                )


if __name__ == "__main__":
    unittest.main()
