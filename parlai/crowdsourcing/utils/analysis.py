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

    @classmethod
    def setup_args(cls):
        parser = argparse.ArgumentParser('Compile crowdsourcing results')
        parser.add_argument(
            '--output-folder', type=str, help='Folder to save output files to'
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):
        self.output_folder = opt.get('output_folder')

    @abstractmethod
    def compile_results(self) -> pd.DataFrame:
        """
        Method for returning the final results dataframe.

        Each row of the dataframe consists of one utterance of one conversation.
        """


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
            '--results-folders', type=str, help='Comma-separated list of result folders'
        )
        parser.add_argument(
            '--problem-buckets',
            type=str,
            help='Comma-separated list of buckets used for annotation',
            default='bucket_0,bucket_1,bucket_2,bucket_3,bucket_4,none_all_good',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        super().__init__(opt)

        # Handle inputs
        if 'results_folders' in opt:
            self.results_folders = opt['results_folders'].split(',')
        else:
            self.results_folders = None
        self.problem_buckets = opt['problem_buckets'].split(',')

        # Validate problem buckets
        if 'none_all_good' not in self.problem_buckets:
            # The code relies on a catchall "none" category if the user selects no other
            # annotation bucket
            raise ValueError(
                'There must be a "none_all_good" category in self.problem_buckets!'
            )
