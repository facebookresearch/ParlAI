#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Configuration file for Fast ACUTE Eval.

CONFIG: Dict[str, Dict]
    - maps ids to their appropriate options
    - for models, please only include options that you would specify on the command line
"""
import os
from typing import Dict

ROOT_DIR = '/checkpoint/parlai/acute_evals/'
CONFIG: Dict[str, Dict] = {
    'example_model_1': {
        'model_file': 'zoo:tutorial_transformer_generator/model',
        'model': 'transformer/generator',
        # general args
        'batchsize': 1,
        'skip_generation': False,
        'interactive_mode': False,
        'beam_size': 3,
        'beam_min_length': 3,
        'inference': 'beam',
        'beam_block_ngram': 3,
        'beam_context_block_ngram': 3,
    },
    'example_model_2': {
        'model_file': 'zoo:tutorial_transformer_generator/model',
        'model': 'transformer/generator',
        # general args
        'batchsize': 1,
        'skip_generation': False,
        'interactive_mode': False,
        'inference': 'nucleus',
        'topp': 0.9,
    },
    'example_model_log': {
        'log_path': f"{os.path.dirname(os.path.realpath(__file__))}/example/chat_log.jsonl"
    },
    'example_dataset': {'task': 'convai2', 'prepended_context': True},
}
