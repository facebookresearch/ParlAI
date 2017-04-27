# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def build(opt):
    dpath = opt['datapath'] + "/MovieDialog/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "movie_dialog_dataset.tgz"
        url = "http://www.thespermwhale.com/jaseweston/babi/" + fname
        build_data.download(dpath, url)
        build_data.untar(dpath, fname)
        dpath2 = dpath + "/movie_dialog_dataset/task4_reddit/"
        fname2a = dpath2 + "p6tyohj"
        fname2b = dpath2 + "p6tyohj.tgz"
        url2 = "http://tinyurl.com/" + "p6tyohj"
        build_data.download(dpath2, url2)
        build_data.move(fname2a, fname2b)
        build_data.untar(dpath2, "p6tyohj.tgz")

        # Mark the data as built.
        build_data.mark_done(dpath)
