#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os


version = '1.0'
URLS = ['https://www.nyu.edu/projects/bowman/multinli/' + 'multinli_' + version + '.zip']
FILE_NAMES = ['multinli_' + version + '.zip']
SHA256 = ['049f507b9e36b1fcb756cfd5aeb3b7a0cfcb84bf023793652987f7e7e0957822']

def build(opt):
    dpath = os.path.join(opt['datapath'], 'MultiNLI')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        # uncompress it
        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
