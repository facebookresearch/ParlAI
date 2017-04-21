#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def buildImage(dpath):
    print("[building image data: " + dpath + "]")
    # download the image data.
    fname1 = "train2014.zip"
    fname2 = "val2014.zip"
    fname3 = "test2014.zip"

    url = "http://msvocds.blob.core.windows.net/coco2014/"

    build_data.download(dpath, url + fname1)
    build_data.download(dpath, url + fname2)
    build_data.download(dpath, url + fname3)
    
    build_data.untar(dpath, fname1, False)
    build_data.untar(dpath, fname2, False)
    build_data.untar(dpath, fname3, False)



def build(opt):
    dpath = opt['datapath'] + "/VQA-COCO2014/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = "Questions_Train_mscoco.zip"
        fname2 = "Questions_Val_mscoco.zip"
        fname3 = "Questions_Test_mscoco.zip"

        fname4 = "Annotations_Val_mscoco.zip"
        fname5 = "Annotations_Train_mscoco.zip"

        url = "http://visualqa.org/data/mscoco/vqa/"
        build_data.download(dpath, url + fname1)
        build_data.download(dpath, url + fname2)
        build_data.download(dpath, url + fname3)

        build_data.download(dpath, url + fname4)
        build_data.download(dpath, url + fname5)

        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)
        build_data.untar(dpath, fname3)
        build_data.untar(dpath, fname4)
        build_data.untar(dpath, fname5)

        buildImage(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath)
        



