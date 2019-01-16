#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'MutualFriends')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        url = (
            'https://worksheets.codalab.org/rest/bundles/'
            '0x5a4cefea7fd443cea15aa532bb8fcd67/contents/blob/'
        )
        build_data.download(url, dpath, 'data.json')

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
