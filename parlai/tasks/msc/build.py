#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.utils.logging import logger
from parlai.core.build_data import DownloadableFile


MSC_DATASETS_VERSION = 'v0.1'


RESOURCES = [
    DownloadableFile(
        f'http://parl.ai/downloads/msc/msc_{MSC_DATASETS_VERSION}.tar.gz',
        f'msc_{MSC_DATASETS_VERSION}.tar.gz',
        'e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60',
    )
]


def get_msc_dir_path(opt):
    dpath = os.path.join(opt['datapath'], 'msc')
    return dpath


def build(opt):
    version = MSC_DATASETS_VERSION
    # create particular instance of dataset depending on flags..
    dpath = get_msc_dir_path(opt)
    if not build_data.built(dpath, version):
        logger.warning('[build data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)
        # Mark the data as built.
        build_data.mark_done(dpath, version)

    return dpath
