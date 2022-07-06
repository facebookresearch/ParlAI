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
        'http://parl.ai/downloads/COCO-IMG/train2014.zip',
        'train2014.zip',
        'f9f102e5336ede4060bb06e1aca438b85f9be18c21960837079c1a88530d498c',
    ),
    DownloadableFile(
        'http://parl.ai/downloads/COCO-IMG/val2014.zip',
        'val2014.zip',
        'e3cb2caf99e37157c48a99883cc8c57eed8ea3942a501c1abf6f7d9c040ddea8',
    ),
    DownloadableFile(
        'http://parl.ai/downloads/COCO-IMG/test2014.zip',
        'test2014.zip',
        'ead40c62230cb2cf70ff4c8b4c70abdc260a7556e77b3282621d06d8e2e35bdf',
    ),
    DownloadableFile(
        'http://parl.ai/downloads/coco_caption/dataset_coco.tgz',
        'dataset_coco.tgz',
        '85fac3c266af928bfec5bbd35f24e2371417f8977350e1de86276455643b09d0',
    ),
]


def buildImage(opt):
    dpath = os.path.join(opt['datapath'], 'COCO-IMG-2014')
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
    dpath = os.path.join(opt['datapath'], 'COCO_2014_Caption')
    version = '1.0'

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        for downloadable_file in RESOURCES[3:]:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
