#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Profile Expanded Attention 1024-truncation model.

Performs two rounds of expanded attention over the full LIGHT context (without dialogue
history).
"""
from parlai.zoo.light_whoami.whoami_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'profile_expanded_attention_1024', 'v1.0')
