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
    dpath = os.path.join(opt["datapath"], "LAUG", opt["augmentation_method"])
    if opt["augmentation_method"].lower() == "nei":
        dpath = dpath = os.path.join(opt["datapath"], "LAUG", "orig")

    # define version if any
    version = "1.0"

    # # check if data had been previously built
    # TODO build the data with CheckDST/data/prepare_multiwoz_dst.sh
    build_data.mark_done(dpath, version_string=version)
