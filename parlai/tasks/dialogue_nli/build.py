#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import parlai.core.build_data as build_data

URLS = ['1WtbXCv3vPB5ql6w0FVDmAEMmWadbrCuG']
FILE_NAMES = ['dialogue_nli.zip']
SHA256 = []

def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'dialogue_nli')
    # define version if any
    version = '1.0'

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256, from_google=True)

        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
