#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Classifier trained on the md_gender task.

For more info see: <parl.ai/projects/md_gender>.
"""

from parlai.core.build_data import download_models


def download(datapath):
    version = 'v1.0'
    fnames = ['md_gender_classifier.tgz']
    download_models(
        opt={'datapath': datapath},
        fnames=fnames,
        model_folder='md_gender',
        version=version,
        use_model_type=False,
    )
