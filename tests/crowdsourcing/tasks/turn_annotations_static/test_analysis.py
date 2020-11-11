#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test components of specific crowdsourcing tasks.
"""

import json
import os
import unittest

import pandas as pd

from parlai.crowdsourcing.tasks.turn_annotations_static.analysis.compile_results import (
    TurnAnnotationsStaticResultsCompiler,
)
import parlai.utils.testing as testing_utils


class TestAnalysis(unittest.TestCase):
    """
    Test the analysis code for the static turn annotations task.
    """

    def test_compile_results(self):
        """
        Test compiling results on a dummy set of data.
        """

        with testing_utils.tempdir() as tmpdir:

            # Define desired stdout

            # Paths
            analysis_config_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'analysis_config'
            )
            temp_gold_annotations_path = os.path.join(tmpdir, 'gold_annotations.json')

            # Save a file of gold annotations
            gold_annotations = {
                "1_0_5": {
                    "bucket_0": False,
                    "bucket_1": False,
                    "bucket_2": False,
                    "bucket_3": False,
                    "bucket_4": False,
                    "none_all_good": True,
                },
                "1_1_5": {
                    "bucket_0": False,
                    "bucket_1": False,
                    "bucket_2": False,
                    "bucket_3": False,
                    "bucket_4": True,
                    "none_all_good": False,
                },
                "2_0_5": {
                    "bucket_0": False,
                    "bucket_1": True,
                    "bucket_2": False,
                    "bucket_3": False,
                    "bucket_4": False,
                    "none_all_good": False,
                },
                "2_1_5": {
                    "bucket_0": False,
                    "bucket_1": False,
                    "bucket_2": False,
                    "bucket_3": False,
                    "bucket_4": True,
                    "none_all_good": False,
                },
            }
            with open(temp_gold_annotations_path, 'w') as f:
                json.dump(gold_annotations, f)

            # Run compilation of results
            opt = {
                'results_folders': os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'analysis_samples'
                ),
                'output_folder': tmpdir,
                'onboarding_in_flight_data_file': os.path.join(
                    analysis_config_folder, 'onboarding_in_flight.jsonl'
                ),
                'gold_annotations_file': temp_gold_annotations_path,
            }
            with testing_utils.capture_output() as output:
                compiler = TurnAnnotationsStaticResultsCompiler(opt)
                compiler.NUM_SUBTASKS = 3
                compiler.NUM_ANNOTATIONS = 2
                compiler.compile_results()
                actual_stdout = output.getvalue()

                # Check the output against what it should be
                desired_stdout = f"""\
I AM OBVIOUSLY WRONG\
"""
            import pdb

            pdb.set_trace()
            self.assertEqual(actual_stdout, desired_stdout)

            # Check that the saved results file is what it should be
            desired_results_path = os.path.join(
                analysis_config_folder, 'desired_results.csv'
            )
            desired_results = pd.read_csv(desired_results_path)
            actual_results_path = [obj for obj in os.listdir(tmpdir)][0]
            actual_results = pd.read_csv(actual_results_path)
            import pdb

            pdb.set_trace()
            self.assertTrue(actual_results.equals(desired_results))


if __name__ == "__main__":
    unittest.main()
