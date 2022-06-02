#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os
from convokit import download

RESOURCES = [
    DownloadableFile(
        'http://zissou.infosci.cornell.edu/convokit/datasets/friends-corpus/friends-corpus.zip',
        'friends-corpus.zip',
        '51ae80ce345212839d256b59b4982e9b40229ff6049115bd54d885a285d2b921',
        zipped=True,
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'Friends')
    version = '1.00'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        # for downloadable_file in RESOURCES:
        #     downloadable_file.download_file(dpath)
        download('friends-corpus', data_dir=dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
