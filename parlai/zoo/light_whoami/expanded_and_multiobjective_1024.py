#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Automated Expanded Attention + Multi-Objective.

Combines both expanded attention and multi-objective training objective
"""
from parlai.zoo.light_whoami.whoami_download import download_with_model_type


def download(datapath):
    download_with_model_type(datapath, 'expanded_and_multiobjective_1024', 'v1.0')
