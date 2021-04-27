#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.tasks.huggingface.agents import AbstractHuggingFaceTeacher


class AxTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'ax'
    hf_text_fields = ['premise', 'hypothesis']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'test', 'valid': 'test', 'test': 'test'}


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


class DefaultTeacher(ColaTeacher):
    pass
