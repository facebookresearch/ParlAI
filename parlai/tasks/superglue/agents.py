#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.tasks.huggingface.agents import AbstractHuggingFaceTeacher


class AxbTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'super_glue'
    hf_name = 'axb'
    hf_text_fields = ['sentence1', 'sentence2']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'test', 'valid': 'test', 'test': 'test'}


class AxgTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'super_glue'
    hf_name = 'axg'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'test', 'valid': 'test', 'test': 'test'}


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


class DefaultTeacher(BoolqTeacher):
    pass
