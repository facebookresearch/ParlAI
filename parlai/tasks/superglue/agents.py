#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import MultiTaskTeacher
from parlai.tasks.huggingface.agents import AbstractHuggingFaceTeacher
from copy import deepcopy


class AxbTeacher(AbstractHuggingFaceTeacher):
    """
    Note: this is an evaluation dataset so it only has a test split
    """

    hf_path = 'super_glue'
    hf_name = 'axb'
    hf_text_fields = ['sentence1', 'sentence2']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': None, 'valid': None, 'test': 'test'}


class AxgTeacher(AbstractHuggingFaceTeacher):
    """
    Note: this is an evaluation dataset so it only has a test split
    """

    hf_path = 'super_glue'
    hf_name = 'axg'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': None, 'valid': None, 'test': 'test'}


class BoolqTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'super_glue'
    hf_name = 'boolq'
    hf_text_fields = ['passage', 'question']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class CbTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'super_glue'
    hf_name = 'cb'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class CopaTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'super_glue'
    hf_name = 'copa'
    hf_text_fields = ['premise', 'question']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class SuperglueTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        superglue_tasks = ['boolq', 'cb', 'copa']
        superglue_tasks = ['superglue:' + t for t in superglue_tasks]
        opt = deepcopy(opt)
        opt['task'] = ', '.join(superglue_tasks)
        super().__init__(opt, shared)


class DefaultTeacher(SuperglueTeacher):
    pass
