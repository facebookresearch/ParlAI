#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl',
        'train.jsonl',
        '58ffa3c8472410e24b8c43f423d89c8a003d8284698a6ed7874355dedd09a2fb',
        zipped=False,
    ),
    DownloadableFile(
        'https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl',
        'dev.jsonl',
        '3210497fdaae614ac085d9eb873dd7f4d49b6f965a93adadc803e1229fd8a02a',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'CommonSenseQA')
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
