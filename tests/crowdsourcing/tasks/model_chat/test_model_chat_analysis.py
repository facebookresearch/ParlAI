#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the analysis code for the model chat task.
"""

import glob
import json
import os
import re
from typing import Any, Dict, List

import pytest
from pytest_regressions.file_regression import FileRegressionFixture

import parlai.utils.testing as testing_utils


try:

    from parlai.crowdsourcing.tasks.model_chat.analysis.compile_results import (
        ModelChatResultsCompiler,
    )
    from parlai.crowdsourcing.utils.tests import check_stdout

    class TestModelChatResultsCompiler(ModelChatResultsCompiler):
        def get_task_data(self) -> List[Dict[str, Any]]:
            fake_jsons = []
            # Load paths
            date_strings = sorted(
                [
                    obj
                    for obj in os.listdir(self.results_folder)
                    if os.path.isdir(os.path.join(self.results_folder, obj))
                    and re.fullmatch(r'\d\d\d\d_\d\d_\d\d', obj)
                ]
            )
            folders = [os.path.join(self.results_folder, str_) for str_ in date_strings]

            for folder in folders:
                for file_name in sorted(os.listdir(folder)):
                    # Read in file
                    with open(os.path.join(folder, file_name), 'rb') as f:
                        data = json.load(f)
                        worker_id = data['workers'][0]
                        assignment_id = data['assignment_ids'][0]
                        fake_jsons.append(
                            {
                                'data': {'save_data': {'custom_data': data}},
                                'worker_id': worker_id,
                                'assignment_id': assignment_id,
                                'status': 'completed',
                            }
                        )

            return fake_jsons

    class TestCompileResults:
        """
        Test the analysis code for the model chat task.
        """

        # Dictionary of cases to test, as well as flags to use with those cases
        CASES = {'basic': '--problem-buckets None', 'with_personas_and_buckets': ''}

        @pytest.fixture(scope="module")
        def setup_teardown(self):
            """
            Call code to set up and tear down tests.

            Run this only once because we'll be running all analysis code before
            checking any results.
            """

            outputs = {}

            for case, flag_string in self.CASES.items():

                # Paths
                analysis_samples_folder = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'analysis_samples', case
                )
                analysis_outputs_folder = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'test_model_chat_analysis',
                )
                outputs[f'{case}__expected_stdout_path'] = os.path.join(
                    analysis_outputs_folder, f'{case}__test_stdout.txt'
                )

                prefixes = ['results', 'worker_results']

                with testing_utils.tempdir() as tmpdir:

                    # Run analysis
                    with testing_utils.capture_output() as output:
                        arg_string = f"""--output-folder {tmpdir} {flag_string}"""
                        parser_ = TestModelChatResultsCompiler.setup_args()
                        args_ = parser_.parse_args(arg_string.split())
                        compiler = TestModelChatResultsCompiler(vars(args_))
                        compiler.results_folder = analysis_samples_folder
                        compiler.compile_and_save_results()
                        stdout = output.getvalue()

                    # Define output structure
                    filtered_stdout = '\n'.join(
                        [
                            line
                            for line in stdout.split('\n')
                            if not line.endswith('.csv')
                        ]
                    )
                    # Don't track lines that record where a file was saved to, because
                    # filenames are timestamped
                    outputs[f'{case}__stdout'] = filtered_stdout
                    for prefix in prefixes:
                        results_path = list(
                            glob.glob(os.path.join(tmpdir, f'{prefix}_*'))
                        )[0]
                        with open(results_path) as f:
                            outputs[f'{case}__{prefix}'] = f.read()

            yield outputs
            # All code after this will be run upon teardown

        def test_stdout(self, setup_teardown):
            """
            Check the output against what it should be.
            """
            outputs = setup_teardown
            for case in self.CASES.keys():
                check_stdout(
                    actual_stdout=outputs[f'{case}__stdout'],
                    expected_stdout_path=outputs[f'{case}__expected_stdout_path'],
                )

        def test_results_file(
            self, setup_teardown, file_regression: FileRegressionFixture
        ):
            """
            Check the results file against what it should be.

            We don't use DataFrameRegression fixture because the results might include
            non-numeric data.
            """
            for case in self.CASES.keys():
                prefix = f'{case}__results'
                outputs = setup_teardown
                file_regression.check(outputs[prefix], basename=prefix)

        def test_worker_results_file(
            self, setup_teardown, file_regression: FileRegressionFixture
        ):
            """
            Check the worker_results file against what it should be.

            We don't use DataFrameRegression fixture because the results might include
            non-numeric data.
            """
            for case in self.CASES.keys():
                prefix = f'{case}__worker_results'
                outputs = setup_teardown
                file_regression.check(outputs[prefix], basename=prefix)

except ImportError:
    pass
