#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://github.com/wise-east/spolin/raw/master/data/spolin-train-acl.json',
        'spolin-train-acl.json',
        'db3e1470f28c9dbf8423c2a8f5e4a52aad932f68a6f86d376597f0f36b845439',
        zipped=False,
    ),
    DownloadableFile(
        'https://github.com/wise-east/spolin/raw/master/data/spolin-train.json',
        'spolin-train.json',
        '3b87ce000d0900a1811298f04ebff1922368f99b68eb9b1d9f436aaf53027f64',
        zipped=False,
    ),
    DownloadableFile(
        'https://github.com/wise-east/spolin/raw/master/data/spolin-valid.json',
        'spolin-valid.json',
        '0db6afb5ddbcbfff974f5ca5f2b575433cc02d3a65662f808e12bd82b634e141',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'spolin')
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
