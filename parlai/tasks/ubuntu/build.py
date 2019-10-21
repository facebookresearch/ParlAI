#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


URLS = ['http://parl.ai/downloads/ubuntu/' + 'ubuntu.tar.gz']
FILE_NAMES = ['ubuntu.tar.gz']
SHA256 = ['87caf297381287247cbff3aa7cc2f4609f3b12c40e8535688168d7413740e0d0']


def build(opt):
    dpath = os.path.join(opt['datapath'], 'Ubuntu')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)
        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
