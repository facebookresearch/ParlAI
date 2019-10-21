#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

URL = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/'

FILE_NAMES = [
    'Questions_Train_mscoco.zip',
    'Questions_Val_mscoco.zip',
    'Questions_Test_mscoco.zip',
    'Annotations_Val_mscoco.zip',
    'Annotations_Train_mscoco.zip',
]

URLS = list(map(lambda x: URL + x, FILE_NAMES))

SHA256 = [
    'c3b2bb6155528eeae95e0a914af394d6f0d98f8f2b51012c44b27778e1a96707',
    'e8839be5de2d711989bf0adc82e6717d1ce307d27c9b1dfb0abf413b79a5d4d0',
    'bd080c297fc863bf8258caa4864d3b5afab29373375a6637f8546338291e28c0',
    '29377c35186d90aeab3e61bdad890f51215d1f88b700bd22ef19004d73bf284f',
    'a5f5f97c162a4ad44896be08bac6deaa258aa3fec281afcc84fe85ae44cb1ebc',
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'VQA-v1')
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
