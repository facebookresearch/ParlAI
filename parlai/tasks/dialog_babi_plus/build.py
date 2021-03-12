#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.core.build_data import DownloadableFile
from parlai.core import build_data

RESOURCES = [
    DownloadableFile(
        'https://drive.google.com/uc?export=download&id=0B2MvoQfXtqZmMTJqclpBdGN2bmc',
        'dialog-bAbI-plus.zip',
        '3a987926ce1b782e9c95771444a98336801741c07ff44bf75bfc8a38fccbdf98',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'dialog-bAbI-plus')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(dpath, version)
