#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


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
        fname1 = 'train2017.zip'
        fname2 = 'val2017.zip'
        fname3 = 'test2017.zip'

        url = 'http://parl.ai/downloads/COCO-IMG/'

        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)
        build_data.download(url + fname3, dpath, fname3)

        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)
        build_data.untar(dpath, fname3)

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
        build_data.make_dir(dpath)

        # download the data.

        fname1 = 'annotations_trainval2017.zip'
        fname2 = 'image_info_test2017.zip'
        # dataset URL
        url = 'http://images.cocodataset.org/annotations/'

        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)

        # uncompress it
        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
