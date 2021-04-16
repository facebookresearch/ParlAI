#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from typing import List
import os
from abc import ABC

# huggingface imports
from datasets import load_dataset


class AbstractHuggingFaceTeacher(DialogTeacher, ABC):
    """
    Abstract parent class for HuggingFace teachers.
    Extend this class and specify the attributes below to use a different dataset.

    hf_path = path parameter passed into hugging face load_dataset function
    hf_name = name parameter passed into hugging face load_dataset function
    hf_text_fields = list of names of the data fields from the dataset to be included in the text/query
    hf_label_field = name of the data field from the hf dataset that specifies the label of the episode
    hf_splits_mapping = dictionary mapping with the keys 'train', 'valid', and 'test', that map to the 
    names of the splits of the hf dataset.
    """

    hf_path: str = 'glue'
    hf_name: str = 'cola'
    hf_text_fields: List[str] = ['sentence']
    hf_label_field: str = 'label'
    hf_splits_mapping: dict = {'train': 'train', 'valid': 'validation', 'test': 'test'}

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.hf_split = self.hf_splits_mapping[self.datatype.split(':')[0]]
        self.data_path = self._path(opt)
        opt['datafile'] = self.data_path

        # load dataset from HuggingFace
        self.dataset = load_dataset(
            path=self.hf_path, name=self.hf_name, split=self.hf_split
        )

        self.id = "huggingface"
        super().__init__(opt, shared)

    def _path(self, opt):
        path = os.path.join(opt['datapath'], 'huggingface')
        return path

    def setup_data(self, path):
        pre_candidates = self.dataset.features[self.hf_label_field].names
        for row in self.dataset:
            # construct text query from the hf_text_fields specified
            text_arr = []
            for col in self.hf_text_fields:
                text_part = row.get(col)
                if text_part is None:
                    raise KeyError(f'Feature "{col}" not found in data.')
                text_arr.append(text_part)
            query = '\n'.join(text_arr)

            # construct label and candidates
            label = row[self.hf_label_field]
            if type(label) is int:
                label = pre_candidates[label]
                candidates = pre_candidates
            if label in row:
                label = row[label]
                candidates = [row[l] for l in pre_candidates]
            yield (query, [label], None, candidates), True


class DefaultTeacher(AbstractHuggingFaceTeacher):
    pass
