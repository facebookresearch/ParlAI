#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import make_dir
from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper
from typing import Dict, Iterable, List, Optional, Tuple
from typing_extensions import TypedDict
import os

# huggingface imports
import datasets


class AbstractHuggingFaceTeacher(DialogTeacher):
    """
    Abstract parent class for HuggingFace teachers. Extend this class and specify the
    attributes below to use a different dataset.

    hf_path = path parameter passed into hugging face load_dataset function
    hf_name = name parameter passed into hugging face load_dataset function
    hf_text_fields = list of names of the data fields from the dataset to be included in the text/query
    hf_message_fields = [optional] list of names of the data fields from the dataset to be included in the message object but *not* text
    hf_label_field = name of the data field from the hf dataset that specifies the label of the episode
    hf_splits_mapping = dictionary mapping with the keys 'train', 'valid', and 'test', that map to the
    names of the splits of the hf dataset.
    render_text_field = bool where if True, will include the text field name in the query (e.g. "sentence: <sentence>")
    """

    def __init__(self, opt, shared=None):
        self.fold = DatatypeHelper.fold(opt['datatype'])
        self.hf_split = self.hf_splits_mapping[self.fold]
        self.data_path = self._path(opt)
        opt['datafile'] = self.data_path
        make_dir(opt['datafile'])

        self.id = "huggingface"
        super().__init__(opt, shared)

    def _path(self, opt):
        if self.hf_name:
            return os.path.join(
                opt['datapath'], 'huggingface', self.hf_path, self.hf_name, self.fold
            )
        return os.path.join(opt['datapath'], 'huggingface', self.hf_path, self.fold)

    def _get_text_value(self, row) -> Tuple[str, Dict[str, str]]:
        """
        return the constructed text query and dict mapping text field names to values.
        """
        # construct text query from the hf_text_fields specified
        text_dict = {}
        for col in self.hf_text_fields:
            text_part = row.get(col)
            if text_part is None:
                raise KeyError(f'Feature "{col}" not found in data.')
            text_dict[col] = text_part
        query = '\n'.join(text_dict.values())
        if hasattr(self, "hf_message_fields"):
            for col in self.hf_message_fields:
                text_part = row.get(col)
                if text_part is None:
                    raise KeyError(f'Feature "{col}" not found in data.')
                text_dict[col] = text_part
        return query, text_dict

    def _get_label_value(self, row):
        """
        return the label value from the data row.
        """
        return row[self.hf_label_field]

    def _get_label_candidates(self, row, label) -> str:
        """
        try to return the true label text value from the row and the candidates.
        """
        if isinstance(self.dataset.features['label'], datasets.features.ClassLabel):
            pre_candidates = self.dataset.features[self.hf_label_field].names
            # construct label and candidates
            if type(label) is int:
                label = pre_candidates[label]
            if label in row:
                return row[label], [row[l] for l in pre_candidates]
            return label, pre_candidates
        else:
            label = str(label)
            return label, [label]

    def setup_data(self, path: str) -> Iterable[tuple]:
        """
        Default implementation of setup_data.

        Manually override if needed.
        """
        # load dataset from HuggingFace
        self.dataset = datasets.load_dataset(
            path=self.hf_path, name=self.hf_name, split=self.hf_split
        )

        for row in self.dataset:
            query, text_dict = self._get_text_value(row)
            label = self._get_label_value(row)
            label, candidates = self._get_label_candidates(row, label)

            episode_dict = text_dict
            episode_dict['text'] = query
            episode_dict['label'] = label
            episode_dict['label_candidates'] = candidates
            yield episode_dict, True


class DefaultTeacher:
    def __init__(self, opt):
        raise NotImplementedError(
            "There is no default teacher for HuggingFace datasets. Please use a specific one."
        )
