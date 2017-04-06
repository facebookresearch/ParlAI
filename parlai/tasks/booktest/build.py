# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def build(opt):
    dpath = opt['datapath'] + "/BookTest/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "booktest.tar.bz2"
        url = "https://s3.amazonaws.com/fair-data/parlai/booktest/" + fname
        build_data.download(dpath, url)
        build_data.untar(dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath)
