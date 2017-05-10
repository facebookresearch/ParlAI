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
    dpath = os.path.join(opt['datapath'], 'VQA-COCO2014')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        for item in os.listdir(dpath):
            item = os.path.join(dpath, item)
            if os.path.isdir(item):
                build_data.remove_dir(item)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = 'Questions_Train_mscoco.zip'
        fname2 = 'Questions_Val_mscoco.zip'
        fname3 = 'Questions_Test_mscoco.zip'

        fname4 = 'Annotations_Val_mscoco.zip'
        fname5 = 'Annotations_Train_mscoco.zip'

        url = 'http://visualqa.org/data/mscoco/vqa/'
        build_data.download(os.path.join(dpath, fname1), url + fname1)
        build_data.download(os.path.join(dpath, fname2), url + fname2)
        build_data.download(os.path.join(dpath, fname3), url + fname3)

        build_data.download(os.path.join(dpath, fname4), url + fname4)
        build_data.download(os.path.join(dpath, fname5), url + fname5)

        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)
        build_data.untar(dpath, fname3)
        build_data.untar(dpath, fname4)
        build_data.untar(dpath, fname5)

        # Mark the data as built.
        build_data.mark_done(dpath)
