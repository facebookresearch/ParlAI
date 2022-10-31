#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"

RESOURCES = [
    # raw data files
    DownloadableFile(
        URL,
        "proofwriter.zip",
        "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26",
        zipped=True,
    ),
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "proof_writer")
    # define version if any
    version = "1.0"

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
