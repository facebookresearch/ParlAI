#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/self_feeding/self_feeding_v031.tar.gz',
        'self_feeding_v031.tar.gz',
        '223d867c72f8b8c173fce86d49d099a56ca002f1a39886c407caee661417a5b4',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'self_feeding')
    version = '3.1'
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(dpath, version)
