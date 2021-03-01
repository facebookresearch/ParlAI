#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the analysis code for the model chat task.
"""

import glob
import os

import pytest
from pytest_regressions.file_regression import FileRegressionFixture

import parlai.utils.testing as testing_utils


try:

    from parlai.crowdsourcing.tasks.model_chat.analysis.compile_results import (
        ModelChatResultsCompiler,
    )
    from parlai.crowdsourcing.utils.tests import check_stdout

    class TestCompileResults:
        """
        Test the analysis code for the model chat task.
        """

        @pytest.fixture(scope="module")
        def setup_teardown(self):
            """
            Call code to set up and tear down tests.

            Run this only once because we'll be running all analysis code before
            checking any results.
            """

            outputs = {}

            # Paths
            analysis_samples_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'analysis_samples'
            )
            analysis_outputs_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'test_model_chat_analysis'
            )
            outputs['expected_stdout_path'] = os.path.join(
                analysis_outputs_folder, 'test_stdout.txt'
            )

            prefixes = ['results', 'worker_results']

            with testing_utils.tempdir() as tmpdir:

                # Run analysis
                with testing_utils.capture_output() as output:
                    arg_string = f"""\
    --results-folders {analysis_samples_folder}
    --output-folder {tmpdir}
    """
                    parser_ = ModelChatResultsCompiler.setup_args()
                    args_ = parser_.parse_args(arg_string.split())
                    ModelChatResultsCompiler(vars(args_)).compile_and_save_results()
                    stdout = output.getvalue()

                # Define output structure
                filtered_stdout = '\n'.join(
                    [line for line in stdout.split('\n') if not line.endswith('.csv')]
                )
                # Don't track lines that record where a file was saved to, because filenames
                # are timestamped
                outputs['stdout'] = filtered_stdout
                for prefix in prefixes:
                    results_path = list(glob.glob(os.path.join(tmpdir, f'{prefix}_*')))[
                        0
                    ]
                    with open(results_path) as f:
                        outputs[prefix] = f.read()

            yield outputs
            # All code after this will be run upon teardown

        def test_stdout(self, setup_teardown):
            """
            Check the output against what it should be.
            """
            outputs = setup_teardown
            check_stdout(
                actual_stdout=outputs['stdout'],
                expected_stdout_path=outputs['expected_stdout_path'],
            )

        def test_results_file(
            self, setup_teardown, file_regression: FileRegressionFixture
        ):
            """
            Check the results file against what it should be.

            We don't use DataFrameRegression fixture because the results might include
            non-numeric data.
            """
            prefix = 'results'
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
            prefix = 'worker_results'
            outputs = setup_teardown
            file_regression.check(outputs[prefix], basename=prefix)


except ImportError:
    pass
