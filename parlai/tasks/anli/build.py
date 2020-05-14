#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile

ANLI_VERSION = 'v0.1'
ANLI = 'ANLI'

RESOURCES = [
    DownloadableFile(
        'https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip',
        'anli_v0.1.zip',
        '16ac929a7e90ecf9093deaec89cc81fe86a379265a5320a150028efe50c5cde8',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], ANLI)
    version = ANLI_VERSION

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
    return dpath, version
