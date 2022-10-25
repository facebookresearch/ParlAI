#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile

import os

"""
eQASC Dataset. Multihop Question-Answering set with explanations.

EMNLP 2020 paper 'Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering'
"""

RESOURCES = [
    DownloadableFile(
        url='1lllyoJLza2k5k8AlDVxJFHSl_pJRzFQb',
        file_name='eqasc_dev_grc.json',
        hashcode='515a34a92c4e69d54939d4b1c3c07fd17eb319d42b7cd0daf7fcfe81f372df8b',
        zipped=False,
        from_google=True,
    ),
    DownloadableFile(
        url='16jyHtTtSV_lb-RVZJZLi-HL8DI73XU0w',
        file_name='eqasc_test_grc.json',
        hashcode='524ecab20b0f635ab1d95d5ce437413c484c6280bfed5806e61c234f1f8c63b8',
        zipped=False,
        from_google=True,
    ),
    DownloadableFile(
        url='13AaSCgbRsqIxvmP-bBZpenvWaAQLs15G',
        file_name='eqasc_train_grc.json',
        hashcode='7603e5c908f4d0baac66f8288b2d6351eaa60d3ded2ea55f9f35b655dbbe0e26',
        zipped=False,
        from_google=True,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'eqasc')
    version = '1.0'

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
