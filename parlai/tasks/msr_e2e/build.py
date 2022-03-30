#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

ROOT_URL = (
    "https://raw.githubusercontent.com/xiul-msr/e2e_dialog_challenge/master/data/"
)

RESOURCES = [
    # raw data files
    DownloadableFile(
        f"{ROOT_URL}/movie_all.tsv",
        "movie_all.tsv",
        "d2291fd898d8c2d92d7c92affa5601a0561a28f07f6147e9c196c5a573a222d6",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/restaurant_all.tsv",
        "restaurant_all.tsv",
        "0e297b2ac2e29f9771fed3cd348873b729eb079cc26f8c2333a28247671bdb28",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/taxi_all.tsv",
        "taxi_all.tsv",
        "6d8ee9719b3d294b558eb53516c897108d1276e9dbcac0101d4e19a2ad801d20",
        zipped=False,
    ),
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "msr_e2e")
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
