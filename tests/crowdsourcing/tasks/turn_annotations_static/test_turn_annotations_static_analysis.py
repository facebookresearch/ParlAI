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

import parlai.utils.testing as testing_utils


try:

    from parlai.crowdsourcing.tasks.turn_annotations_static.analysis.compile_results import (
        TurnAnnotationsStaticResultsCompiler,
    )
    from parlai.crowdsourcing.utils.tests import check_stdout

    class TestAnalysis(unittest.TestCase):
        """
        Test the analysis code for the static turn annotations task.
        """

        def test_compile_results(self):
            """
            Test compiling results on a dummy set of data.
            """

            with testing_utils.tempdir() as tmpdir:

                # Define expected stdout

                # Paths
                analysis_samples_folder = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'analysis_samples'
                )
                analysis_outputs_folder = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'test_turn_annotations_static_analysis',
                )
                expected_stdout_path = os.path.join(
                    analysis_outputs_folder, 'test_stdout.txt'
                )
                temp_gold_annotations_path = os.path.join(
                    tmpdir, 'gold_annotations.json'
                )

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
                parser = TurnAnnotationsStaticResultsCompiler.setup_args()
                parser.set_defaults(
                    **{
                        'results_folders': analysis_samples_folder,
                        'output_folder': tmpdir,
                        'onboarding_in_flight_data_file': os.path.join(
                            analysis_samples_folder, 'onboarding_in_flight.jsonl'
                        ),
                        'gold_annotations_file': temp_gold_annotations_path,
                    }
                )
                args = parser.parse_args([])
                with testing_utils.capture_output() as output:
                    compiler = TurnAnnotationsStaticResultsCompiler(vars(args))
                    compiler.NUM_SUBTASKS = 3
                    compiler.NUM_ANNOTATIONS = 3
                    compiler.compile_and_save_results()
                    actual_stdout = output.getvalue()

                # Check the output against what it should be
                check_stdout(
                    actual_stdout=actual_stdout,
                    expected_stdout_path=expected_stdout_path,
                )

                # Check that the saved results file is what it should be
                sort_columns = ['hit_id', 'worker_id', 'conversation_id', 'turn_idx']
                expected_results_path = os.path.join(
                    analysis_outputs_folder, 'expected_results.csv'
                )
                expected_results = (
                    pd.read_csv(expected_results_path)
                    .drop('folder', axis=1)
                    .sort_values(sort_columns)
                    .reset_index(drop=True)
                )
                # Drop the 'folder' column, which contains a system-dependent path
                # string
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
                if not actual_results.equals(expected_results):
                    raise ValueError(
                        f'\n\n\tExpected results:\n{expected_results.to_csv()}'
                        f'\n\n\tActual results:\n{actual_results.to_csv()}'
                    )

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
