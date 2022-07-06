#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained polyencoder retrieval model trained on the ConvAI2, EmpatheticDialogues, and
Wizard of Wikipedia dialogue tasks and fine-tuned on BlendedSkillTalk.
"""

from parlai.core.build_data import download_models


def download(datapath):
    model_type = 'multi_task_bst_tuned'
    version = 'v1.0'
    opt = {'datapath': datapath, 'model_type': model_type}
    fnames = [f'{version}.tar.gz']
    download_models(
        opt=opt,
        fnames=fnames,
        model_folder='blended_skill_talk',
        version=version,
        use_model_type=True,
    )
