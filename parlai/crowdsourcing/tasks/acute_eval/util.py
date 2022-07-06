#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.dataframe_regression import DataFrameRegressionFixture
from pytest_regressions.file_regression import FileRegressionFixture

from parlai.crowdsourcing.utils.tests import AbstractOneTurnCrowdsourcingTest


TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def get_hashed_combo_path(
    root_dir: str,
    subdir: str,
    task: str,
    combos: Iterable[Union[List[str], Tuple[str, str]]],
) -> str:
    """
    Return a unique path for the given combinations of models.

    :param root_dir: root save directory
    :param subdir: immediate subdirectory of root_dir
    :param task: the ParlAI task being considered
    :param combos: the combinations of models being compared
    """

    # Sort the names in each combo, as well as the overall combos
    sorted_combos = []
    for combo in combos:
        assert len(combo) == 2
        sorted_combos.append(tuple(sorted(combo)))
    sorted_combos = sorted(sorted_combos)

    os.makedirs(os.path.join(root_dir, subdir), exist_ok=True)
    path = os.path.join(
        root_dir,
        subdir,
        hashlib.sha1(
            '___and___'.join(
                [f"{m1}vs{m2}.{task.replace(':', '_')}" for m1, m2 in sorted_combos]
            ).encode('utf-8')
        ).hexdigest()[:10],
    )
    return path


class AbstractFastAcuteTest(AbstractOneTurnCrowdsourcingTest):
    """
    Abstract test class for testing Fast ACUTE code.
    """

    TASK_DIRECTORY = TASK_DIRECTORY
    MODELS = ['model1', 'model2']
    MODEL_STRING = ','.join(MODELS)
    TASK_DATA = {
        "final_data": [
            {"speakerChoice": "human_as_model", "textReason": "Makes more sense"},
            {"speakerChoice": "model1", "textReason": "Makes more sense"},
            {"speakerChoice": "model2", "textReason": "Makes more sense"},
            {"speakerChoice": "model1", "textReason": "Makes more sense"},
            {"speakerChoice": "model2", "textReason": "Makes more sense"},
        ]
    }

    def _get_common_overrides(self, root_dir: str) -> List[str]:
        """
        Return overrides for all subclassed Fast ACUTE test code.
        """

        return [
            'mephisto.blueprint.block_on_onboarding_fail=False',
            f'mephisto.blueprint.onboarding_path={self.TASK_DIRECTORY}/task_config/onboarding.json',
            f'mephisto.blueprint.root_dir={root_dir}',
            'mephisto.task.task_name=acute_eval_test',
        ]

    def test_agent_state(self, setup_teardown, data_regression: DataRegressionFixture):
        outputs = setup_teardown
        self._check_agent_state(state=outputs['state'], data_regression=data_regression)

    def test_all_convo_pairs_txt(
        self, setup_teardown, file_regression: FileRegressionFixture
    ):
        outputs = setup_teardown
        self._check_file_contents(
            results_folder=outputs['results_folder'],
            file_suffix='all_convo_pairs.txt',
            file_regression=file_regression,
        )

    def test_all_html(self, setup_teardown, file_regression: FileRegressionFixture):
        outputs = setup_teardown
        self._check_file_contents(
            results_folder=outputs['results_folder'],
            file_suffix='all.html',
            file_regression=file_regression,
        )

    def test_full_csv(
        self, setup_teardown, dataframe_regression: DataFrameRegressionFixture
    ):
        outputs = setup_teardown
        self._check_dataframe(
            results_folder=outputs['results_folder'],
            file_suffix='full.csv',
            dataframe_regression=dataframe_regression,
            drop_columns=['task_start', 'time_taken'],
        )

    def test_grid_csv(
        self, setup_teardown, dataframe_regression: DataFrameRegressionFixture
    ):
        outputs = setup_teardown
        self._check_dataframe(
            results_folder=outputs['results_folder'],
            file_suffix='grid.csv',
            dataframe_regression=dataframe_regression,
        )

    def test_grid_winners_as_rows_csv(
        self, setup_teardown, dataframe_regression: DataFrameRegressionFixture
    ):
        outputs = setup_teardown
        self._check_dataframe(
            results_folder=outputs['results_folder'],
            file_suffix='grid.winners_as_rows.csv',
            dataframe_regression=dataframe_regression,
        )

    def test_ratings_per_worker_csv(
        self, setup_teardown, dataframe_regression: DataFrameRegressionFixture
    ):
        outputs = setup_teardown
        self._check_dataframe(
            results_folder=outputs['results_folder'],
            file_suffix='ratings_per_worker.csv',
            dataframe_regression=dataframe_regression,
        )

    def test_reason_html(self, setup_teardown, file_regression: FileRegressionFixture):
        outputs = setup_teardown
        self._check_file_contents(
            results_folder=outputs['results_folder'],
            file_suffix='reason.html',
            file_regression=file_regression,
        )

    def test_significance_csv(
        self, setup_teardown, dataframe_regression: DataFrameRegressionFixture
    ):
        outputs = setup_teardown
        self._check_dataframe(
            results_folder=outputs['results_folder'],
            file_suffix='significance.csv',
            dataframe_regression=dataframe_regression,
        )

    def _check_dataframe(
        self,
        results_folder: str,
        file_suffix: str,
        dataframe_regression: DataFrameRegressionFixture,
        drop_columns: Optional[list] = None,
    ):
        """
        Check a dataframe for correctness.

        Check the dataframe stored at the file with suffix file_suffix in the folder
        results_folder. Pass in drop_column to indicate columns that shouldn't be
        checked because they will vary across runs (for instance, timestamp columns).
        """
        if drop_columns is None:
            drop_columns = []
        file_path = self._get_matching_file_path(
            results_folder=results_folder, file_suffix=file_suffix
        )
        df = pd.read_csv(file_path).drop(columns=drop_columns)
        dataframe_regression.check(data_frame=df)

    def _check_file_contents(
        self,
        results_folder: str,
        file_suffix: str,
        file_regression: FileRegressionFixture,
    ):
        file_path = self._get_matching_file_path(
            results_folder=results_folder, file_suffix=file_suffix
        )
        with open(file_path) as f:
            # correct whitespace to deal with fbcode demanding disk files not
            # have many trailing newlines
            contents = f.read().rstrip('\n') + '\n'
        file_regression.check(contents=contents)

    def _get_matching_file_path(self, results_folder: str, file_suffix: str) -> str:
        matching_files = [
            obj for obj in os.listdir(results_folder) if obj.endswith(file_suffix)
        ]
        assert len(matching_files) == 1
        return os.path.join(results_folder, matching_files[0])
