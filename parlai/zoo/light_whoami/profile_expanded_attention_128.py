#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Profile Expanded Attention 128-truncation model.

Performs two rounds of expanded attention over the character names and the self persona.
"""
from parlai.zoo.light_whoami.whoami_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'profile_expanded_attention_128', 'v1.0')
