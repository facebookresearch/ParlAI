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
        'http://parl.ai/downloads/COCO-IMG/train2017.zip',
        'train2017.zip',
        '69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929',
    ),
    DownloadableFile(
        'http://parl.ai/downloads/COCO-IMG/val2017.zip',
        'val2017.zip',
        '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05',
    ),
    DownloadableFile(
        'http://parl.ai/downloads/COCO-IMG/test2017.zip',
        'test2017.zip',
        'c7908c3c9f94ba2f3340ebbeec58c25db6be8774f18d68c2f15d0e369d95baba',
    ),
    DownloadableFile(
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'annotations_trainval2017.zip',
        '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268',
    ),
    DownloadableFile(
        'http://images.cocodataset.org/annotations/image_info_test2017.zip',
        'image_info_test2017.zip',
        'e52f412dd7195ac8f98d782b44c6dd30ea10241e9f42521f67610fbe055a74f8',
    ),
]


def buildImage(opt):
    dpath = os.path.join(opt['datapath'], 'COCO-IMG-2017')
    version = '1'

    if not build_data.built(dpath, version_string=version):
        print('[building image data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the image data.
        for downloadable_file in RESOURCES[:3]:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)


def build(opt):
    dpath = os.path.join(opt['datapath'], 'COCO_2017_Caption')
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(os.path.join(dpath, 'annotations'))
        # Make the subfolder into which files will be unzipped

        # download the data.
        for downloadable_file in RESOURCES[3:]:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
