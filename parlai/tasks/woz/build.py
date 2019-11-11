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
        'https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_test_en.json',
        'woz_test_en.json',
        '3673e433b21a6b0d74e9144bd059e64b29bc3e1c5dc0e18654a98ec597c0d72c',
        zipped=False,
    ),
    DownloadableFile(
        'https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_train_en.json',
        'woz_train_en.json',
        '7cd9e971553e5f3e80bb0c93164bf4c619c7f49f45d636a0512474cdeb074485',
        zipped=False,
    ),
    DownloadableFile(
        'https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_validate_en.json',
        'woz_validate_en.json',
        'ae1ea9067fd05c0179d349f140b38de1b2db587d5bfcb4f99ef0d77474ab00ad',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'WoZ')
    version = 'None'

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
