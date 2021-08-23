#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import parlai.core.build_data as build_data
import parlai.utils.logging as logging

import parlai.tasks.wizard_of_internet.constants as CONST


DATASET_FILE = build_data.DownloadableFile(
    'http://parl.ai/downloads/wizard_of_internet/wizard_of_internet.tgz',
    'wizard_of_internet.tgz',
    'c2495b13ad00015e431d51738e02d37d2e80c8ffd6312f1b3d273dd908a8a12c',
)


def build(opt):
    dpath = os.path.join(opt['datapath'], CONST.DATASET_NAME)
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
