#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Classifier trained on dialogue safety datasets: WikiToxicComments, Build it Break it Fix
it standard and adversarial tasks, Bot Adversarial Dialogue tasks (dialogue history
truncated at length 4)
"""

from parlai.core.build_data import download_models


def download(datapath):
    version = 'v1'
    model_type = 'multi_turn'
    opt = {'datapath': datapath, 'model_type': model_type}
    fnames = [f'models_{version}.tar.gz']
    download_models(
        opt=opt,
        fnames=fnames,
        model_folder='bot_adversarial_dialogue',
        version=version,
        use_model_type=True,
    )
