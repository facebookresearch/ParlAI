#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-d-train.json',
        'train.json',
        'baf81f327dee84c6f451c9a4dd662e6193c67473b8791ffb72cce75cdb528f20',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-d-test.json',
        'test.json',
        'e9920491b31f9d00ecf31e51727b495dd6b0d05f4a96f273a343e81b6775a8f0',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-d-dev.json',
        'dev.json',
        '8c7054930a40aeb288ad7c51c42fa93d54aef678ccab29c75d46a7432f4f6278',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'C3')
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
