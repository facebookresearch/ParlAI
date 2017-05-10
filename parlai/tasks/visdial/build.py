# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def buildImage(opt):
    dpath = os.path.join(opt['datapath'], 'COCO-IMG')

    if not build_data.built(dpath):
        print('[building image data: ' + dpath + ']')
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the image data.
        fname1 = 'train2014.zip'
        fname2 = 'val2014.zip'
        fname3 = 'test2014.zip'

        url = 'http://msvocds.blob.core.windows.net/coco2014/'

        build_data.download(dpath, url + fname1)
        build_data.download(dpath, url + fname2)
        build_data.download(dpath, url + fname3)

        build_data.untar(dpath, fname1, False)
        build_data.untar(dpath, fname2, False)
        build_data.untar(dpath, fname3, False)

        # Mark the data as built.
        build_data.mark_done(dpath)


def build(opt):
    dpath = os.path.join(opt['datapath'], 'VisDial-v0.9')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = 'visdial_0.9_train.zip'
        fname2 = 'visdial_0.9_val.zip'

        url = 'https://computing.ece.vt.edu/~abhshkdz/data/visdial/'
        build_data.download(dpath, url + fname1)
        build_data.download(dpath, url + fname2)


        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)

        # Mark the data as built.
        build_data.mark_done(dpath)
