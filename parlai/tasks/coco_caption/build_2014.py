# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

from parlai.tasks.vqa_v1.build import buildImage  # imported by coco_caption.agents


def build(opt):
    dpath = os.path.join(opt['datapath'], 'COCO_2014_Caption')
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

        fname1 = 'annotations_trainval2014.zip'
        fname2 = 'image_info_test2014.zip'
        # dataset URL
        url = 'http://images.cocodataset.org/annotations/'

        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)

        # uncompress it
        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
