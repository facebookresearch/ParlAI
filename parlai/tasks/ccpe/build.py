# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

URLS = ['https://storage.googleapis.com/dialog-data-corpus/CCPE-M-2019/data.json']
FILE_NAMES = ['ccpe.json']
SHA256 = ['14abc40f5ab93eb68607454968f0e3af21aeb75d8c37b8b19bf9eeb957907a42']

def build(opt):
    dpath = os.path.join(opt['datapath'], 'CCPE')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
