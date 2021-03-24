#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from typing import Optional
import os

# huggingface imports
import datasets
from datasets import load_dataset, concatenate_datasets


class HuggingFaceTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.data_path = HuggingFaceTeacher._path(opt)
        opt['datafile'] = self.data_path

        # load dataset from HuggingFace
        self.dataset = load_dataset(
            path=opt['hf_path'],
            name=opt['hf_name'],
            cache_dir=self.data_path,
            split=opt['hf_split'],
        )

        if opt['hf_split'] is None and isinstance(
            self.dataset, datasets.dataset_dict.DatasetDict
        ):
            # no split specified and there are splits- combine all the splits together
            self.dataset = concatenate_datasets(list(self.dataset.values()))

        # map numerical labels back to strings
        self.labels = self.dataset.features['label'].names
        # self.labels.append('unknown')

        def convert_to_str_label(row):
            row['label'] = self.labels[row['label']]
            return row

        self.dataset = self.dataset.map(convert_to_str_label)

        self.id = "huggingface"
        super().__init__(opt, shared)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('Hugging Face Teacher Arguments')
        parser.add_argument(
            '-hfp',
            '--hf_path',
            type=str,
            default='glue',
            help="HuggingFace dataset identifier name",
        )
        parser.add_argument(
            '-hfn',
            '--hf_name',
            type=str,
            default='cola',
            help="defining the name of the HuggingFace dataset configuration",
        )
        parser.add_argument(
            '-hfs',
            '--hf_split',
            type=str,
            default=None,
            help="which split of the HuggingFace dataset to load",
        )
        return parser

    def _path(opt):
        path = os.path.join(opt['datapath'], 'huggingface')
        return path

    def setup_data(self, path):
        for row in self.dataset:
            premise = row.get('premise', '')
            hypothesis = row.get('hypothesis', '')
            sentence = row.get('sentence', '')
            text = sentence + premise + hypothesis
            label = str(row['label'])
            candidates = self.labels

            yield (text, [label], None, candidates), True


class DefaultTeacher(HuggingFaceTeacher):
    pass
