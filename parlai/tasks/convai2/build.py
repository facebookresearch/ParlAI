#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

URLS = ['http://parl.ai/downloads/convai2/convai2_fix_723.tgz']
FILE_NAMES = ['convai2_fix_723.tgz']
SHA256 = ['d0ae89defe2fd0b0a4221eaa642a457d7d40cef475f54798119c7f3b8dd9361d']


def build_fb_format():
    pass


def build(opt):
    version = 'v5.0'
    dpath = os.path.join(opt['datapath'], 'ConvAI2')

    if not build_data.built(dpath, version):
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
        build_data.mark_done(dpath, version)
