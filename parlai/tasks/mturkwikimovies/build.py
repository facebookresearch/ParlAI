#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import parlai.tasks.wikimovies.build as wikimovies_build


def build(opt):
    # Depends upon another dataset, wikimovies, build that first.
    wikimovies_build.build(opt)

    dpath = opt['datapath'] + "/MTurkWikiMovies/"
    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "mturkwikimovies.tar.gz"
        url = ("https://s3.amazonaws.com/fair-data/parlai/mturkwikimovies/"
               + fname)
        build_data.download(dpath, url)
        build_data.untar(dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath)
