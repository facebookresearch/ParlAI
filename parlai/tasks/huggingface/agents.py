#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper
from typing import Any, Dict, Iterable, List, Optional, Tuple
from typing_extensions import TypedDict
from abc import ABC
import os

# huggingface imports
from datasets import load_dataset


class SplitsMappingDict(TypedDict):
    train: str
    valid: str
    test: str


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

    hf_path: str
    hf_name: Optional[str]
    hf_text_fields: List[str]
    hf_label_field: str
    hf_splits_mapping: SplitsMappingDict

    def __init__(self, opt, shared=None):
        self.fold = DatatypeHelper.fold(opt['datatype'])
        self.hf_split = self.hf_splits_mapping[self.fold]
        if self.hf_name:
            opt['datafile'] = os.path.join(
                opt['datapath'], 'huggingface', self.hf_path, self.hf_name, self.fold
            )
        else:
            opt['datafile'] = os.path.join(
                opt['datapath'], 'huggingface', self.hf_path, self.fold
            )

        self.id = "huggingface"
        super().__init__(opt, shared)

    @property
    def hf_path(self) -> str:
        raise NotImplementedError

    @property
    def hf_name(self) -> Optional[str]:
        return None

    @property
    def hf_text_fields(self) -> List[str]:
        raise NotImplementedError

    @property
    def hf_label_field(self) -> str:
        raise NotImplementedError

    @property
    def hf_splits_mapping(self) -> SplitsMappingDict:
        raise NotImplementedError

    def setup_data(self, path: str) -> Iterable[tuple]:
        """
        Default implementation of setup_data.

        Manually override if needed.
        """

        def _get_text_value(row) -> Tuple[str, Dict[str, str]]:
            """
            return the constructed text query and dict mapping text field names to
            values.
            """
            # construct text query from the hf_text_fields specified
            text_dict = {}
            for col in self.hf_text_fields:
                text_part = row.get(col)
                if text_part is None:
                    raise KeyError(f'Feature "{col}" not found in data.')
                text_dict[col] = text_part
            return '\n'.join(text_dict.values()), text_dict

        def _get_label_value(row):
            return row[self.hf_label_field]

        def _get_label_candidates(row, label) -> str:
            pre_candidates = dataset.features[self.hf_label_field].names
            # construct label and candidates
            if type(label) is int:
                return pre_candidates[label], pre_candidates
            if label in row:
                return row[label], [row[l] for l in pre_candidates]
            return label, pre_candidates

        # load dataset from HuggingFace
        dataset = load_dataset(
            path=self.hf_path, name=self.hf_name, split=self.hf_split
        )

        for row in dataset:
            query, text_dict = _get_text_value(row)
            label = _get_label_value(row)
            label, candidates = _get_label_candidates(row, label)

            episode_dict = text_dict
            episode_dict['text'] = query
            episode_dict['label'] = label
            episode_dict['label_candidates'] = candidates
            yield episode_dict, True


class GlueColaTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'cola'
    hf_text_fields = ['sentence']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class DefaultTeacher(AbstractHuggingFaceTeacher):
    def __init__():
        raise NotImplementedError(
            "There is no default teacher for HuggingFace datasets. Please use a specific one."
        )
