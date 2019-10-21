#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

URLS = ["1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA"]
FILE_NAMES = ['qangaroo.zip']
SHA256 = ['2f512869760cdad76a022a1465f025b486ae79dc5b8f0bf3ad901a4caf2d3050']


def build(opt):
    dpath = os.path.join(opt['datapath'], 'qangaroo')
    version = 'v1.1'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256, from_google=True)
        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
