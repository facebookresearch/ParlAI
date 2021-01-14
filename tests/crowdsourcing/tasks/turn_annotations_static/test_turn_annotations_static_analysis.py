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

            # Define expected stdout

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
            parser = TurnAnnotationsStaticResultsCompiler.setup_args()
            parser.set_defaults(
                **{
                    'results_folders': os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'analysis_samples'
                    ),
                    'output_folder': tmpdir,
                    'onboarding_in_flight_data_file': os.path.join(
                        analysis_config_folder, 'onboarding_in_flight.jsonl'
                    ),
                    'gold_annotations_file': temp_gold_annotations_path,
                }
            )
            args = parser.parse_args([])
            with testing_utils.capture_output() as output:
                compiler = TurnAnnotationsStaticResultsCompiler(vars(args))
                compiler.NUM_SUBTASKS = 3
                compiler.NUM_ANNOTATIONS = 3
                compiler.compile_results()
                actual_stdout = output.getvalue()

            # Check the output against what it should be
            expected_stdout = f"""\
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
            for expected_line in expected_stdout.split('\n'):
                if not any(
                    expected_line in actual_line for actual_line in actual_stdout_lines
                ):
                    raise ValueError(
                        f'\n\tThe following line:\n\n{expected_line}\n\n\twas not found '
                        f'in the actual stdout:\n\n{actual_stdout}'
                    )

            # Check that the saved results file is what it should be
            sort_columns = ['hit_id', 'worker_id', 'conversation_id', 'turn_idx']
            expected_results_path = os.path.join(
                analysis_config_folder, 'expected_results.csv'
            )
            expected_results = (
                pd.read_csv(expected_results_path)
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
            if not actual_results.equals(expected_results):
                raise ValueError(
                    f'\n\n\tExpected results:\n{expected_results.to_csv()}'
                    f'\n\n\tActual results:\n{actual_results.to_csv()}'
                )


if __name__ == "__main__":
    unittest.main()
