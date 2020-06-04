#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://storage.googleapis.com/airdialogue/airdialogue_data.tar.gz',
        'airdialogue.tar.gz',
        '4ebf1e7e44078fa0a1986539f6d1827851982420045c674ff26d869e02560b05',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'])
    airdialogue_path = os.path.join(dpath, 'airdialogue_data')
    version = '1.0'

    if not build_data.built(airdialogue_path, version_string=version):
        print('[building data: ' + airdialogue_path + ']')
        if build_data.built(airdialogue_path):
            build_data.remove_dir(airdialogue_path)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(airdialogue_path, version_string=version)
