#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

DECODE = 'decode'
DECODE_PREFIX = 'decode_'
DECODE_VERSION = 'v0.1'


RESOURCES = [
    DownloadableFile(
        'https://sharenlpfile-01.s3.amazonaws.com/data/decode_v0.1.zip',
        'decode_v0.1.zip',
        # '0badc03c41813ae9748f259370ce655e576d736fea2d084dd6a786ac59f2f2a1',
        '084aab98652a04ce4a78c1a63d91575f5ab416a0c474b962ca1f4508a56b7484',
        True,
        False,
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], DECODE)

    version = DECODE_VERSION

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
