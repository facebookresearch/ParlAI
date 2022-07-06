#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/nlpdata/dream/master/data/train.json',
        'train.json',
        '90942ddea1b56231a0ad2097dc5f115ce1face1cfb86f029041e5bad38e68566',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/nlpdata/dream/master/data/test.json',
        'test.json',
        'd96d7d0752f7eab1ea8f165a582430f35653c428d36633ab4d7f26fc14946c3a',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/nlpdata/dream/master/data/dev.json',
        'dev.json',
        '9d5af2e580d809c73872a7dd43fe93d0b07c6f6086b04a9a9a1917603009d961',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'DREAM')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(dpath, version_string=version)
