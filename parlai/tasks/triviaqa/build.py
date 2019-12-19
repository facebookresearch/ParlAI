#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz',
        'triviaqa-rc.tar.gz',
        'ef94fac6db0541e5bb5b27020d067a8b13b1c1ffc52717e836832e02aaed87b9',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'TriviaQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
