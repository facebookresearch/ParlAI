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
        "fd1c36de494c4fabb790d03894b4b4c7",
        'empatheticdialogues.tar.gz',
        'daf91fd2f8c2c9bc1713c74257b0a8643a0c78a87a26f946cceb5278f3445b1d',
        from_clearml=True,
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'ClearMLData')
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
