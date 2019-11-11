#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/dbll/dbll.tgz',
        'dbll.tgz',
        'd8c727dac498b652c7f5de6f72155dce711ff46c88401a303399d3fad4db1e68',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'DBLL')
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

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
