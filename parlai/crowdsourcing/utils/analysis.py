#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
from abc import ABC, abstractmethod
from datetime import datetime
import json
from typing import Any, Dict, List, Union

import pandas as pd

from parlai.core.opt import Opt
import parlai.utils.logging as logging

# Defining the class only if Mephisto is installed, since it relies on Mephisto
try:
    from mephisto.abstractions.databases.local_database import LocalMephistoDB
    from mephisto.data_model.unit import Unit
    from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
except ImportError:
    pass


class AbstractResultsCompiler(ABC):
    """
    Abstract class for compiling results of crowdsourcing runs.

    Currently only provides utility attributes/methods for analyzing turn annotations.
    """

    @classmethod
    def setup_args(cls):
        parser = argparse.ArgumentParser('Compile crowdsourcing results')
        parser.add_argument(
            '--output-folder', type=str, help='Folder to save output files to'
        )
        parser.add_argument(
            '--results-format',
            type=str,
            choices=['csv', 'json'],
            default='csv',
            help='Output format for results data',
        )
        parser.add_argument(
            '--task-name', type=str, help='Name of the Mephisto task to open'
        )
        parser.add_argument(
            '--database-path',
            type=str,
            default=None,
            help='Path to local Mephisto database. Leave empty for default location.',
        )
        return parser

    def __init__(self, opt: Opt):
        self.task_name = opt['task_name']
        self.output_folder = opt['output_folder']
        self.results_format = opt.get('results_format', 'json')
        self.database_path = opt['database_path']

        # We lazily load these later, or inject their mock version during testing.
        self._mephisto_db = None
        self._mephisto_data_browser = None

    def get_mephisto_data_browser(self) -> MephistoDataBrowser:
        if not self._mephisto_data_browser:
            db = self.get_mephisto_db()
            self._mephisto_data_browser = MephistoDataBrowser(db=db)
        return self._mephisto_data_browser

    def get_mephisto_db(self) -> LocalMephistoDB:
        if not self._mephisto_db:
            self._mephisto_db = LocalMephistoDB(self.database_path)
        return self._mephisto_db

    def get_results_path_base(self) -> str:
        """
        Return the save path for the results file, not including the file extension.
        """
        now = datetime.now()
        return os.path.join(
            self.output_folder,
            f'{self.__class__.__name__}__{now.strftime("%Y%m%d_%H%M%S")}',
        )

    def get_worker_name(self, worker_id: str) -> str:
        """
        Gets the global id of a worker from their Mephisto worker_id.

        The worker_id is the unique id that the crowdsourcing platforms (eg, Amazon
        Mechanical Turk) assign to a single human worker in their system.
        """
        db = self.get_mephisto_db()
        return db.get_worker(worker_id)["worker_name"]

    def get_task_units(self) -> List[Unit]:
        """
        Retrieves the list of work units from the Mephisto task.
        """
        data_browser = self.get_mephisto_data_browser()
        return data_browser.get_units_for_task_name(self.task_name)

    def get_data_from_unit(self, unit: Unit) -> Dict[str, Any]:
        """
        Retrieves task data for a single unit.
        """
        try:
            data_browser = self.get_mephisto_data_browser()
            return data_browser.get_data_from_unit(unit)
        except (IndexError, AssertionError) as error:
            logging.error(error)
            logging.warning(
                f'Skipping unit {unit.db_id}. No message found for this unit.'
            )

    def get_task_data(self) -> List[Dict[str, Any]]:
        """
        Retrieves task data for a list of Mephisto task units.
        """
        task_data = []
        for unit in self.get_task_units():
            unit_data = self.get_data_from_unit(unit)
            if unit_data and self.is_unit_acceptable(unit_data):
                task_data.append(unit_data)

        return task_data

    def is_unit_acceptable(self, unit_data: Dict[str, Any]) -> bool:
        """
        Helps filtering units that are compiled. Override for use.

        Returning False means that the unit data will be discarded.
        """
        if not unit_data:
            # Add your task-specific qualificaiton logic that justifies
            # discarding this unit, based on it data content.
            return False

        return True

    @abstractmethod
    def compile_results(self) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Method for returning the final results as a dataframe or a json.

        For Dict output each key is a unique identifier (eg Assignment ID) for a unit of
        crowdsourcing work. The data for that unit is stored in the value as dictionary.

        Each row of the dataframe consists of one utterance of one conversation, or crowdsourcing interaction.

        NOTE: Preference for new projects is Dict output (see the TODO below).
        TODO: Only support Dict. Deprecate ` pd.DataFrame` when no other code is relying on it.
        """

    def _validate_compiled_result_type(self, results):
        assert isinstance(results, dict) or isinstance(results, pd.DataFrame), (
            'The output of result compiler needs to be a dictionary or a pandas dataframe. '
            f'Found ({type(results)})'
        )

    def compile_and_save_results(self):
        """
        Compile results and save them.

        Results will be saved in the format given by --results-format.
        """
        compiled_results = self.compile_results()
        self._validate_compiled_result_type(compiled_results)
        results_path_base = self.get_results_path_base()
        results_path = f'{results_path_base}.{self.results_format}'
        os.makedirs(self.output_folder, exist_ok=True)
        if self.results_format == 'csv':
            if not isinstance(compiled_results, pd.DataFrame):
                logging.warning(
                    "The requested data output format was 'csv' while the data was compiled as a 'dict'. "
                    'Transforming dictionary data into pd.DataFrame using pandas.'
                )
                compiled_results = pd.DataFrame.from_dict(
                    compiled_results, orient='index'
                )
            compiled_results.to_csv(results_path, index=False)
        elif self.results_format == 'json':
            if isinstance(compiled_results, pd.DataFrame):
                logging.warning(
                    "The requested data output format was 'json' while the data was compiled as a 'dataframe'. "
                    'Transforming dataframe into json using pandas.'
                )
                # Reset the index to make each row have a unique index value
                compiled_results.reset_index().to_json(results_path)
            else:
                with open(results_path, 'w') as fout:
                    fout.write(json.dumps(compiled_results))

        else:
            raise ValueError(
                f'Results save format of "{self.results_format}" currently unsupported!'
            )
        logging.info(f'Wrote results file to {results_path}.')


class AbstractTurnAnnotationResultsCompiler(AbstractResultsCompiler):
    """
    Results compiler subclass to provide utility code for turn annotations.

    Currently incompatible with Mephisto's DataBrowser: all subclasses load results
    files directly from disk.
    TODO: make all subclasses compatible with DataBrowser
    """

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--problem-buckets',
            type=str,
            help='Comma-separated list of buckets used for annotation. Set to an empty string to not analyze problem buckets.',
            default='bucket_0,bucket_1,bucket_2,bucket_3,bucket_4,none_all_good',
        )
        return parser

    def __init__(self, opt: Opt):

        super().__init__(opt)

        # Handle inputs
        if opt['problem_buckets'].lower() not in ['', 'none']:
            self.use_problem_buckets = True
            self.problem_buckets = opt['problem_buckets'].split(',')
        else:
            self.use_problem_buckets = False
            self.problem_buckets = []
