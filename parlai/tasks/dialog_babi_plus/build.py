#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.core.build_data import DownloadableFile
from parlai.core import build_data

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/dialog_babi_plus/dialog-babi-plus-v1.tgz',
        'dialog-babi-plus-v1.tgz',
        'c3ca01e970a607d8ad01a47420d6b493c43c9ca70211bfe15b01a461437b950e',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'dialog-bAbI-plus')
    version = "v1.1"

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
