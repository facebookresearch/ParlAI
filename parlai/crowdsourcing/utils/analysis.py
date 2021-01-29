#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.data_model.unit import Unit
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser


class AbstractResultsCompiler(ABC):
    """
    Abstract class for compiling results of crowdsourcing runs.

    Currently only provides utility attributes/methods for analyzing turn annotations.
    """

    @classmethod
    def setup_args(cls):
        parser = argparse.ArgumentParser('Compile crowdsourcing results')
        parser.add_argument(
            '--results-folders', type=str, help='Comma-separated list of result folders'
        )
        parser.add_argument(
            '--output-folder', type=str, help='Folder to save output files to'
        )
        parser.add_argument(
            '--problem-buckets',
            type=str,
            help='Comma-separated list of buckets used for annotation',
            default='bucket_0,bucket_1,bucket_2,bucket_3,bucket_4,none_all_good',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        # Handle inputs
        if 'results_folders' in opt:
            self.results_folders = opt['results_folders'].split(',')
        else:
            self.results_folders = None
        self.output_folder = opt.get('output_folder')
        self.problem_buckets = opt['problem_buckets'].split(',')

        # Validate problem buckets
        if 'none_all_good' not in self.problem_buckets:
            # The code relies on a catchall "none" category if the user selects no other
            # annotation bucket
            raise ValueError(
                'There must be a "none_all_good" category in self.problem_buckets!'
            )

    @abstractmethod
    def compile_results(self) -> pd.DataFrame:
        """
        Method for returning the final results dataframe.
        """


class AbstractDataBrowserResultsCompiler(AbstractResultsCompiler):
    @classmethod
    def setup_args(cls):
        parser = argparse.ArgumentParser('Compile crowdsourcing results')
        parser.add_argument(
            '--task-name', type=str, help='Name of the Mephisto task to open'
        )
        return parser

    def __init__(self, opt):
        self.task_name = opt["task_name"]
        self._mephisto_data_browser = None

    def get_mephisto_data_browser(self) -> MephistoDataBrowser:
        if not self._mephisto_data_browser:
            db = LocalMephistoDB()
            self._mephisto_data_browser = MephistoDataBrowser(db=db)
        return self._mephisto_data_browser

    def get_task_units(self, task_name: str) -> List[Unit]:
        """
        Retrieves the list of work units from the Mephisto task.
        """
        data_browser = self.get_mephisto_data_browser()
        return data_browser.get_units_for_task_name(task_name)

    def get_units_data(self, task_units: List[Unit]) -> List[Dict]:
        """
        Retrieves task data for a lits of Mephisto task units.
        """
        data_browser = self.get_mephisto_data_browser()
        task_data = []
        for unit in task_units:
            task_data.append(data_browser.get_data_from_unit(unit))
        return task_data
