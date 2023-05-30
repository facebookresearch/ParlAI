#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import parlai.utils.logging as logging


DATASET_NAME = 'parlai_multilight'
DATASET_FILE = build_data.DownloadableFile(
    f'http://parl.ai/downloads/projects/multilight/{DATASET_NAME}.tar.gz',
    f'{DATASET_NAME}.tar.gz',
    'cbc20e4fa7a551c0efec4a4129e75335d3f3586797d6f767e320403079f4a6b2',
)


def build(opt):
    dpath = opt['datapath']
    version = '1.0'
    if not build_data.built(dpath, version):
        logging.info(
            f'[building data: {dpath}]\nThis may take a while but only heppens once.'
        )
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        DATASET_FILE.download_file(dpath)
        logging.info('Finished downloading dataset files successfully.')

        build_data.mark_done(dpath, version)
