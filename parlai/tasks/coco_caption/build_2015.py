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
        'http://parl.ai/downloads/COCO-IMG/test2015.zip',
        'test2015.zip',
        '52d45179bbe4fcc41ba16550b3df532fc9d0a0084b2afaeb6a3ae396032aaf14',
    ),
    DownloadableFile(
        'http://images.cocodataset.org/annotations/image_info_test2015.zip',
        'image_info_test2015.zip',
        'cf400242f8497257fb8a3e369bc766491f4a7e42625fb3d72555504e9a8c3b18',
    ),
]


def buildImage(opt):
    dpath = os.path.join(opt['datapath'], 'COCO-IMG-2015')
    version = '1'

    if not build_data.built(dpath, version_string=version):
        print('[building image data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES[:1]:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)


def build(opt):
    dpath = os.path.join(opt['datapath'], 'COCO_2015_Caption')
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES[1:]:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
