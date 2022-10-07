#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys, os

import parlai.core.build_data as build_data

from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y",
        "MULTIWOZ2.1.zip",
        "d377a176f5ec82dc9f6a97e4653d4eddc6cad917704c1aaaa5a8ee3e79f63a8e",
    )
]


def build(opt):
    # get path to data directory
    datapath = opt["datapath"]
    env_datapath = os.environ.get("DATAPATH", "")
    if env_datapath:
        datapath = env_datapath
    dpath = os.path.join(datapath, "multiwoz_dst")
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

        # download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # # # TODO: reformat multiwoz

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
