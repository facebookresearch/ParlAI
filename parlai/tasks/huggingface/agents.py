#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from typing import Optional
import os
from abc import ABC

# huggingface imports
from datasets import load_dataset


class AbstractHuggingFaceTeacher(DialogTeacher, ABC):
    """
    Abstract parent class for HuggingFace teachers.
    Extend this class and specify the attributes below to use a different dataset.
    """

    hf_path = 'glue'
    hf_name = 'cola'
    hf_text_fields = ['sentence']
    hf_label_field = 'label'

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.hf_split = self.datatype.split(':')[0]
        if self.hf_split == 'valid':
            self.hf_split = 'validation'
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
            # construct text
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
