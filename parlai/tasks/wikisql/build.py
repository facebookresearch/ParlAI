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
        'https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2',
        'data.tar.bz2',
        '755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881',
    ),
    DownloadableFile(
        'https://github.com/salesforce/WikiSQL/raw/master/lib/query.py',
        'query.py',
        'f539150bea6cd07a5dca226abcced6f9d356d216f5c3d70107693613f1fbeb25',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'WikiSQL')
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
