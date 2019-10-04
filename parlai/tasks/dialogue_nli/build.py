#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import parlai.core.build_data as build_data


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'dialogue_nli')
    # define version if any
    version = '1.0'

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        fname = 'dialogue_nli.zip'
        gd_id = '1WtbXCv3vPB5ql6w0FVDmAEMmWadbrCuG'
        build_data.download_from_google_drive(gd_id, os.path.join(dpath, fname))

        # uncompress it
        build_data.unzip(dpath, fname)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
