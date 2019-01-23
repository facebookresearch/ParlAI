#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'qangaroo')
    version = 'v1.1'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'qangaroo.zip'
        g_ID = "1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA"

        print("downloading ...")
        build_data.download_from_google_drive(
            g_ID, os.path.join(dpath, fname))
        build_data.untar(dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
