#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://github.com/Alab-NII/onecommon/archive/master.zip',
        'onecommon.zip',
        'a07eb871653e5d5d53c554b85e1824f8ae1493e4472c60a54c634c2a2d7d5626',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'onecommon')
    version = None

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
