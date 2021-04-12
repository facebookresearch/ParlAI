#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://download.microsoft.com/download/E/B/8/EB84CB1A-D57D-455F-B905-3ABDE80404E5/metalwoz-v1.zip',
        'metalwoz-v1.zip',
        '2a2ae3b25760aa2725e70bc6480562fa5d720c9689a508d28417631496d6764f',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'metalwoz')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        build_data.make_dir(os.path.join(dpath, 'dialogues'))

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(dpath, version_string=version)
