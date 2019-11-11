#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz',
        'wizard_of_wikipedia.tgz',
        '2a549627a83fea745efa2076a41d1c0078ad002ab2b54eae6a4e3d3d66ae24b7',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'wizard_of_wikipedia')
    version = '1.0'
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(dpath, version)
