#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'CNN_DM')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.

        cnn_fname = 'cnn_stories.tgz'
        cnn_gd_id = '0BwmD_VLjROrfTHk4NFg2SndKcjQ'
        build_data.download_from_google_drive(cnn_gd_id, os.path.join(dpath, cnn_fname))
        build_data.untar(dpath, cnn_fname)

        dm_fname = 'dm_stories.tgz'
        dm_gd_id = '0BwmD_VLjROrfM1BxdkxVaTY2bWs'
        build_data.download_from_google_drive(dm_gd_id, os.path.join(dpath,  dm_fname))
        build_data.untar(dpath, dm_fname)


        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)


