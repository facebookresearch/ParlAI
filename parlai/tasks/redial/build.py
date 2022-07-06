#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://github.com/ReDialData/website/raw/data/redial_dataset.zip',
        'redial.zip',
        'b48756681ec6f84e0af36979c5e9baa21ea8d9e7036b8764ea9b787bb0baf69b',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'redial')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        # Download the data.
        downloadable_file = RESOURCES[0]
        downloadable_file.download_file(dpath)
        build_data.mark_done(dpath, version_string=version)
