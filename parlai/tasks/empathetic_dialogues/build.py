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
        'http://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz',
        'empatheticdialogues.tar.gz',
        '56f234d77b7dd1f005fd365bb17769cfe346c3c84295b69bc069c8ccb83be03d',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'empatheticdialogues')
    version = '1.0'

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
