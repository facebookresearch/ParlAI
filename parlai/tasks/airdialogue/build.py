#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
import shutil
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://storage.googleapis.com/airdialogue/airdialogue_data.tar.gz',
        'airdialogue.tar.gz',
        '4ebf1e7e44078fa0a1986539f6d1827851982420045c674ff26d869e02560b05',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'airdialogue')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Re-organize the directory to be less redundant
        print('reorganizing airdialogue directory')
        actual_data_dir = os.path.join(dpath, 'airdialogue_data', 'airdialogue')
        resources_dir = os.path.join(dpath, 'airdialogue_data', 'resources')
        readme = os.path.join(dpath, 'airdialogue_data', 'readme.txt')
        shutil.move(actual_data_dir, dpath)
        shutil.move(resources_dir, dpath)
        shutil.move(readme, dpath)
        os.rmdir(os.path.join(dpath, 'airdialogue_data'))

        build_data.mark_done(dpath, version_string=version)
