#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

URLS = [
    'https://rajpurkar.github.io/SQuAD-explorer/dataset/' + 'train-v1.1.json',
    'https://rajpurkar.github.io/SQuAD-explorer/dataset/' + 'dev-v1.1.json',
    'http://parl.ai/downloads/squad-fulldocs/' + 'squad_fulldocs.tgz',
]
FILE_NAMES = ['train-v1.1.json', 'dev-v1.1.json', 'squad_fulldocs.tgz']
SHA256 = [
    '3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955',
    '95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9',
    '199fbe66524270bc8423e5d788267ef6ac5029e12443428430e080f3c057b534',
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'SQuAD')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS[:2], FILE_NAMES[:2], SHA256[:2])

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

    if 'fulldoc' in opt['task']:
        dpath += '-fulldoc'
        if not build_data.built(dpath, version_string=version):
            print('[building data: ' + dpath + ']')
            if build_data.built(dpath):
                # An older version exists, so remove these outdated files.
                build_data.remove_dir(dpath)
            build_data.make_dir(dpath)

            # Download the data.
            build_data.download_check(dpath, URLS[2:], FILE_NAMES[2:], SHA256[2:])
            for zipfile in FILE_NAMES[2:]:
                build_data.untar(dpath, zipfile)

            # Mark the data as built.
            build_data.mark_done(dpath, version_string=version)
