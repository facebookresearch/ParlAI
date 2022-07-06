#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/dstc7/dstc7_v2.tgz',
        'dstc7_v2.tgz',
        'cc8fd830f9894768ab4f7b104cddd4105456812ab614041337ec12c5a3a56685',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'dstc7')
    version = '2.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
