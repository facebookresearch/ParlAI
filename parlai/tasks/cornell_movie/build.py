#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager
import parlai.core.build_data as build_data
import codecs
import os

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/cornell_movie/cornell_movie_dialogs_corpus.tgz',
        'cornell_movie_dialogs_corpus.tgz',
        'ae77ab2e4743ce929087a4f529934059b920c4bdaa3143741b65b1e648ab45fd',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'CornellMovie')
    version = 'v1.01'

    if not build_data.built(dpath, version_string=version):
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
