#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'VQA-v1')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = 'Questions_Train_mscoco.zip'
        fname2 = 'Questions_Val_mscoco.zip'
        fname3 = 'Questions_Test_mscoco.zip'

        fname4 = 'Annotations_Val_mscoco.zip'
        fname5 = 'Annotations_Train_mscoco.zip'

        url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/'
        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)
        build_data.download(url + fname3, dpath, fname3)
        build_data.download(url + fname4, dpath, fname4)
        build_data.download(url + fname5, dpath, fname5)

        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)
        build_data.untar(dpath, fname3)
        build_data.untar(dpath, fname4)
        build_data.untar(dpath, fname5)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
