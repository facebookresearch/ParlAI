#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

url = 'https://dada.cs.washington.edu/qasrl/data/'
FILE_NAMES = ['wiki1.train.qa', 'wiki1.dev.qa', 'wiki1.test.qa']
URLS = list(map(lambda x: url + x, FILE_NAMES))
SHA256 = [
    'f927417e94e67b7ae17e33dd882989a5556d7ff37376f8bf5c78ece7d17a6c64',
    'caa94beaaf22304422cdc1a2fd8732b1a47401c9555a81e1f4da81e0a7557a8b',
    'b43a998344fbd520955fb8f0f7b3691ace363daa8628552cf5cf5c8d84df6cca',
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'QA-SRL')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
