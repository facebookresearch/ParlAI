#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os

URLS = ['https://github.com/facebookresearch/end-to-end-negotiator/archive/master.zip']
FILE_NAMES = ['negotiation.zip']
SHA256 = ['0f62af6ced9d0c41183118ccce4ef012886fce7ccbe23565280c5f0da358a2e5']


def build(opt):
    dpath = os.path.join(opt['datapath'], 'negotiation')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data from github
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)

        # Mark as done
        build_data.mark_done(dpath, version_string=version)
