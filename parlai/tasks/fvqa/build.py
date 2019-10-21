#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os

URLS = ['https://dl.dropboxusercontent.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip']
FILE_NAMES = ['FVQA.zip']
SHA256 = ['66d1831a61d1282fb0c95c01435eda9b465961d507c1e166e4c32b89687c3c26']


def build(opt):
    dpath = os.path.join(opt['datapath'], 'FVQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        # An older version exists, so remove these outdated files.
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
