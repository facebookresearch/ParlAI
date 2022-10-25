#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile

import os

RESOURCES = [
    DownloadableFile(
        url='1kVr-YsUVFisceiIklvpWEe0kHNSIFtNh',
        file_name='entailment_trees_emnlp2021_data_v3.zip',
        hashcode='fe05a02f181bb3d27fa2f8bafda824f7a988af9df59f848d694458925be7c497',
        zipped=True,
        from_google=True,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'entailment_bank')
    version = '1.0'

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
