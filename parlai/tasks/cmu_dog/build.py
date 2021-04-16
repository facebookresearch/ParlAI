#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/cmu_dog/cmu_dog.tar.gz',
        'cmu_dog.tar.gz',
        '30d2bac0dae6b4e4c0b94ba581fffaf1acb46838480f7ad6736ad03d9312ae9d',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'cmu_dog')
    version = '1.1'
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
