#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
FILE_NAMES = ['train-v2.0.json', 'dev-v2.0.json']
URLS = list(map(lambda x: URL + x, FILE_NAMES))
SHA256 = [
    '68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002',
    '80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8',
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'SQuAD2')
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
