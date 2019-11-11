#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/openai/generating-reviews-discovering-sentiment/master/data/test_binary_sent.csv',
        'test_binary_sent.csv',
        '519cea7ed4d22fe7ec4eccbb3d5ba6d88902a3b15ce129f476aa8364463a9fc7',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/openai/generating-reviews-discovering-sentiment/master/data/train_binary_sent.csv',
        'train_binary_sent.csv',
        '6003623bcb35aad3a446a265b8931b7ccab61fcc10f2e9c1fec916ff67c7be35',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/openai/generating-reviews-discovering-sentiment/master/data/dev_binary_sent.csv',
        'dev_binary_sent.csv',
        'f34c4987fea208fefc2d62a1b42c83a766cbfc7ce58c2a878ef953cf91f01729',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'SST')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
