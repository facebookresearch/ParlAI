#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os


RESOURCES = [
    DownloadableFile(
        'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip',
        'CLEVR_v1.0.zip',
        '5cd61cf1096ed20944df93c9adb31e74d189b8459a94f54ba00090e5c59936d1',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'CLEVR')
    version = 'v1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        # An older version exists, so remove these outdated files.
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
