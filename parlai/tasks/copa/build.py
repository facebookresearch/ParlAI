#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

RESOURCES = [
    DownloadableFile(
        'http://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz',
        'COPA-resources.tgz',
        '5145348834d2081ad90da0397d1db3d70fa044e506bd8ce224194d24b04cdbbe',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'COPA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
