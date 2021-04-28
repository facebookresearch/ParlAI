#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import MultiTaskTeacher
from parlai.tasks.huggingface.agents import AbstractHuggingFaceTeacher
from copy import deepcopy


class AxTeacher(AbstractHuggingFaceTeacher):
    """
    Note: this is an evaluation dataset so it only has a test split
    Use a model trained on MulitNLI to produce predictions for this dataset.
    """

    hf_path = 'glue'
    hf_name = 'ax'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': None, 'valid': None, 'test': 'test'}


class ColaTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'cola'
    hf_text_fields = ['sentence']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class MnliTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'mnli'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {
        'train': 'train',
        'valid': 'validation_matched',
        'test': 'test_matched',
    }


class MnliMatchedTeacher(AbstractHuggingFaceTeacher):
    """
    Note: this is an evaluation dataset so it only has valid and test splits
    """

    hf_path = 'glue'
    hf_name = 'mnli_matched'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': None, 'valid': 'validation', 'test': 'test'}


class MnliMismatchedTeacher(AbstractHuggingFaceTeacher):
    """
    Note: this is an evaluation dataset so it only has valid and test splits
    """

    hf_path = 'glue'
    hf_name = 'mnli_mismatched'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': None, 'valid': 'validation', 'test': 'test'}


class MrpcTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'mrpc'
    hf_text_fields = ['sentence1', 'sentence2']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class QnliTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'qnli'
    hf_text_fields = ['sentence', 'question']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class QqpTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'qqp'
    hf_text_fields = ['question1', 'question2']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class RteTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'rte'
    hf_text_fields = ['sentence1', 'sentence2']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class Sst2Teacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'sst2'
    hf_text_fields = ['sentence']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class StsbTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'stsb'
    hf_text_fields = ['sentence1', 'sentence2']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class WnliTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'wnli'
    hf_text_fields = ['sentence1', 'sentence2']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class GlueTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        glue_tasks = [
            'cola',
            'mnli',
            'mrpc',
            'qnli',
            'qqp',
            'rte',
            'sst2',
            'stsb',
            'wnli',
        ]
        glue_tasks = ['glue:' + t for t in glue_tasks]
        opt = deepcopy(opt)
        opt['task'] = ', '.join(glue_tasks)
        super().__init__(opt, shared)


class DefaultTeacher(GlueTeacher):
    pass
