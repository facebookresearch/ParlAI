#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/kushalchawla/CaSiNo/main/data/split/casino_train.json',
        'casino_train.json',
        '6b953d153fc8c78f27e911c1439b93b9b3519357e3ba825091b2e567845ba3a7',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/kushalchawla/CaSiNo/main/data/split/casino_valid.json',
        'casino_valid.json',
        '91f2d1f09accedf98667ac081fd5083752738390734e991601b036643da077e0',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/kushalchawla/CaSiNo/main/data/split/casino_test.json',
        'casino_test.json',
        'bf6da2d7c105396300d85a65819c04d99304ac9abb8a590ba342fd0c86b4dd12',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'casino')
    version = "v1.1"

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark as done
        build_data.mark_done(dpath, version_string=version)
