#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import os

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://drive.google.com/u/1/uc?id=1d-987AjyfVdFnohBqQ3yaHt-b5mVezHa&export=download&confirm=t',
        'prosocial_dialog_v1.tar.gz',
        '112e402e283949cbc36b67a86877c8aea098a7fe40fd3e180095e1e147958eba',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'prosocial_dialog')
    version = '0.1'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
