#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os

URLS = [
    'https://storage.googleapis.com/dialog-data-corpus/TASKMASTER-1-2019/'
    + 'self-dialogs.json',
    'https://storage.googleapis.com/dialog-data-corpus/TASKMASTER-1-2019/'
    + 'woz-dialogs.json',
]
FILE_NAMES = ['self-dialogs.json', 'woz-dialogs.json']
SHA256 = [
    '1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d',
    'cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f',
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'taskmaster-1')
    # define version if any
    version = "1.0"

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
