#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

RESOURCES = [
    DownloadableFile(
        'https://github.com/facebookresearch/end-to-end-negotiator/archive/bbb93bbf00f69fced75d5c0d22e855bda07c9b78.zip',
        'negotiation.zip',
        '101f1ce90c3d86a55b097821de812af8e747004beb7a763a63127545b178ddf4',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'negotiation')
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
