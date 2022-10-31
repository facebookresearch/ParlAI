#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"

RESOURCES = [
    # raw data file
    DownloadableFile(
        URL,
        "ASDiv.xml",
        "ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929",
        zipped=False,
    ),
]

VERSION = "1.0"


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "asdiv")
    # define version if any
    version = VERSION

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
