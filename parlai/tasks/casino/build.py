#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/kushalchawla/CaSiNo/main/data/casino.json',
        'casino.json',
        '4f2c4560a0070906ed018c3f0766e35f8f8f31b36274ebf35b608621915744ab',
        zipped=False,
    )
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
