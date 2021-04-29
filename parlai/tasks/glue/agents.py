#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.tasks.huggingface.agents import AbstractHuggingFaceTeacher


class ColaTeacher(AbstractHuggingFaceTeacher):
    hf_path = 'glue'
    hf_name = 'cola'
    hf_text_fields = ['sentence']
    hf_label_field = 'label'
    hf_splits_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}


class DefaultTeacher(ColaTeacher):
    pass
