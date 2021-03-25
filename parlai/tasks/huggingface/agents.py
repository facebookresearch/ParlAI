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

        self.labels = self.dataset.features['label'].names

        # identify text columns
        self.text_columns = []
        text_attr = [
            'premise',
            'hypothesis',
            'sentence',
            'context',
            'question',
            'title',
            'tokens',
        ]
        print(self.dataset.column_names)
        for col in self.dataset.column_names:
            for a in text_attr:
                if a in col:
                    self.text_columns.append(col)
                    break

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
            # construct text
            text_arr = []
            for col in self.text_columns:
                text_arr.append(row.get(col))
            text = '\n'.join(text_arr)

            # construct label and candidates
            label = row['label']
            candidates = self.labels
            if type(label) is int:
                label = self.labels[label]
            if label in row:
                label = row[label]
                candidates = [row[l] for l in self.labels]

            yield (text, [label], None, candidates), True


class DefaultTeacher(HuggingFaceTeacher):
    pass