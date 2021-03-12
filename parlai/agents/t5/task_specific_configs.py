#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Task-specific generation configs for T5.

Taken from HF: https://huggingface.co/t5-small/resolve/main/config.json
"""

TASK_CONFIGS = {
    "summarization": {
        "early_stopping": True,
        "length_penalty": 2.0,
        "max_length": 200,
        "min_length": 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "prefix": "summarize: ",
    },
    "translation_en_to_de": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to German: ",
    },
    "translation_en_to_fr": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to French: ",
    },
    "translation_en_to_ro": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to Romanian: ",
    },
}
