#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class AbstractResultsCompiler(ABC):
    """
    Abstract class for compiling results of crowdsourcing runs.

    Currently only provides utility attributes/methods for analyzing turn annotations.
    """

    PROBLEM_BUCKETS = [
        'bucket_0',
        'bucket_1',
        'bucket_2',
        'bucket_3',
        'bucket_4',
        'none_all_good',
    ]

    @classmethod
    def setup_args(cls):
        parser = argparse.ArgumentParser('Compile crowdsourcing results')
        parser.add_argument(
            '--results-folders',
            type=str,
            help='Comma-separated list of result folders (example: "/basefolder/mephisto/data/runs/NO_PROJECT/123")',
        )
        parser.add_argument(
            '--output-folder', type=str, help='Folder to save output files to'
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        # Handle inputs
        if 'results_folders' in opt:
            self.results_folders = opt['results_folders'].split(',')
        else:
            self.results_folders = None
        self.output_folder = opt.get('output_folder')

        # Validate problem buckets
        if 'none_all_good' not in self.PROBLEM_BUCKETS:
            raise ValueError(
                'There must be a "none_all_good" category in the problem buckets!'
            )

    @abstractmethod
    def compile_results(self) -> pd.DataFrame:
        """
        Method for returning the final results dataframe.
        """
