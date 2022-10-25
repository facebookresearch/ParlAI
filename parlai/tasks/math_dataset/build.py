#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

URL = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"

RESOURCES = [
    # raw data file
    DownloadableFile(
        URL,
        "MATH.tar",
        "0fbe4fad0df66942db6c221cdcc95b298cc7f4595a2f0f518360cce84e90d9ac",
        zipped=True,
    ),
]

VERSION = "1.0"


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"])
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
