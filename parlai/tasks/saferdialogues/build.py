#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

SAFERDIALOGUES_VERSION = "v0.3"

RESOURCES = [
    DownloadableFile(
        f'http://parl.ai/downloads/saferdialogues/saferdialogues_{SAFERDIALOGUES_VERSION}.tar.gz',
        f'saferdialogues_{SAFERDIALOGUES_VERSION}.tar.gz',
        '3d1bc731fb0c63d9f61a52b3d5f1aab0911cda1ca38f49822c79c18a2dc8e834',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'saferdialogues')
    version = SAFERDIALOGUES_VERSION

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
