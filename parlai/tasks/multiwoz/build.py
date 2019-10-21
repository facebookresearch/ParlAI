#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os

URLS = ['https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y']
FILE_NAMES = ['MULTIWOZ2.1.zip']
SHA256 = ['d377a176f5ec82dc9f6a97e4653d4eddc6cad917704c1aaaa5a8ee3e79f63a8e']

def build(opt):
    dpath = os.path.join(opt['datapath'], 'multiwoz')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        # uncompress it
        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)
            
        build_data.mark_done(dpath, version_string=version)
