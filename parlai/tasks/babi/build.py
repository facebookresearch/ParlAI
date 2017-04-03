#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def build(opt):
    dpath = opt['datapath'] + "/bAbI/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "tasks_1-20_v1-2.tar.gz"
        url = "http://www.thespermwhale.com/jaseweston/babi/" + fname
        build_data.download(dpath, url)
        build_data.untar(dpath, fname)

        # Mark the data as built.
        # TODO(@jase): should only do this if the above operations succeeded
        build_data.mark_done(dpath)
