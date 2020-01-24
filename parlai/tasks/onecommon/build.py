#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://github.com/Alab-NII/onecommon/archive/v1.0.zip',
        'onecommon.zip',
        '9e6b7e71ca5baa7528c95ce007afaff31c94e057e88d84ee1f2ff908bfdb2519',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'onecommon')
    version = "1.0"

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
