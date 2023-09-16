#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.


import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import parlai.utils.logging as logging

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/bb3x/bb3_demo_chunks.tgz',
        'bb3_demo_chunks.tgz',
        'a901e121f336504fa559992d9196124642fb8e1cd682bc7216f26abdb0d4900e',
    )
]


def build(opt):
    version = 'v1.0'
    dpath = os.path.join(opt['datapath'], 'bb3_demo')

    if not build_data.built(dpath, version):
        logging.info('building data: ' + dpath)
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
