#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper
from typing import List
from abc import ABC

# huggingface imports
from datasets import load_dataset


class AbstractHuggingFaceTeacher(DialogTeacher, ABC):
    """
    Abstract parent class for HuggingFace teachers. Extend this class and specify the
    attributes below to use a different dataset.

    hf_path = path parameter passed into hugging face load_dataset function
    hf_name = name parameter passed into hugging face load_dataset function
    hf_text_fields = list of names of the data fields from the dataset to be included in the text/query
    hf_label_field = name of the data field from the hf dataset that specifies the label of the episode
    hf_splits_mapping = dictionary mapping with the keys 'train', 'valid', and 'test', that map to the
    names of the splits of the hf dataset.
    render_text_field = bool where if True, will include the text field name in the query (e.g. "sentence: <sentence>")
    """

    hf_path: str = 'glue'
    hf_name: str = 'cola'
    hf_text_fields: List[str] = ['sentence']
    hf_label_field: str = 'label'
    hf_splits_mapping: dict = {'train': 'train', 'valid': 'validation', 'test': 'test'}
    render_text_field: bool = False

    def __init__(self, opt, shared=None):
        self.fold = DatatypeHelper.fold(opt['datatype'])
        self.hf_split = self.hf_splits_mapping[self.fold]
        opt['datafile'] = self.hf_split

        self.id = "huggingface"
        super().__init__(opt, shared)

    def setup_data(self, split):
        """
        Default implementation of setup_data.

        Manually override if needed.
        """
        # load dataset from HuggingFace
        dataset = load_dataset(path=self.hf_path, name=self.hf_name, split=split)

        pre_candidates = dataset.features[self.hf_label_field].names
        for row in dataset:
            # construct text query from the hf_text_fields specified
            text_arr = []
            for col in self.hf_text_fields:
                text_part = row.get(col)
                if text_part is None:
                    raise KeyError(f'Feature "{col}" not found in data.')
                if self.render_text_field:
                    text_part = col + ': ' + text_part
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
