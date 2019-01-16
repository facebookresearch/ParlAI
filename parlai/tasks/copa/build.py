#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'COPA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        fname = 'COPA-resources.tgz'
        # dataset URL
        url = 'http://people.ict.usc.edu/~gordon/downloads/' + fname
        build_data.download(url, dpath, fname)

        # uncompress it
        build_data.untar(dpath, fname)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
