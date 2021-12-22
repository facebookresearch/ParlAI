#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RPA Re-Ranker Model (for automated expanded attention)
"""
from parlai.zoo.light_whoami.whoami_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'rpa_reranker_auto_expanded_attention', 'v1.0')
