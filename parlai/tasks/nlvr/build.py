#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://github.com/clic-lab/nlvr/archive/master.zip',
        'nlvr.zip',
        '32694f83835bd28b86b0f2734efa9544401ed18bd954649b50d1375d43e56b8b',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'nlvr')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark as done
        build_data.mark_done(dpath, version_string=version)
