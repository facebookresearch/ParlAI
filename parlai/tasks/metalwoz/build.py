#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        "https://download.microsoft.com/download/E/B/8/EB84CB1A-D57D-455F-B905-3ABDE80404E5/metalwoz-v1.zip",
        "metalwoz-v1.zip",
        "2a2ae3b25760aa2725e70bc6480562fa5d720c9689a508d28417631496d6764f",
    ),
    DownloadableFile(
        "https://download.microsoft.com/download/0/c/4/0c4a8893-cbf9-4a43-a44a-09bab9539234/metalwoz-test-v1.zip",
        "metalwoz-test-v1.zip",
        "6722d1d9ec05334dd801972767ae3bdefcd15f71bf73fea1d098f214a96a7c6c",
    ),
]


def build(opt):
    dpath = os.path.join(opt["datapath"], "metalwoz")
    version = "1.0"

    if not build_data.built(dpath, version_string=version):
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        build_data.make_dir(os.path.join(dpath, "train", "dialogues"))
        build_data.make_dir(os.path.join(dpath, "test", "dialogues"))

        # Download the data.
        RESOURCES[0].download_file(os.path.join(dpath, "train"))
        RESOURCES[1].download_file(os.path.join(dpath, "test"))

        build_data.untar(os.path.join(dpath, "test"), "dstc8_metalwoz_heldout.zip")
        build_data.mark_done(dpath, version_string=version)
