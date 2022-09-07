#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.utils.logging import logger
from parlai.core.build_data import DownloadableFile

VERSION = 'v0.1'


def _get_dpath_by_task_version(dpath, opt):
    if opt['fits_task_version'] == 'v1' and opt.get('unseen_task', False):
        return os.path.join(dpath, 'human_model_chats_unseen')

    if opt['fits_task_version'] == 'v1':
        return os.path.join(dpath, 'human_model_chats')
    if opt['fits_task_version'] == 'v2':
        return os.path.join(dpath, 'human_model_chats_v2')


RESOURCES = [
    DownloadableFile(
        f'http://parl.ai/downloads/fits/fits_data_{VERSION}.tar.gz',
        f'fits_data_{VERSION}.tar.gz',
        'fe85ab0b24dd904360cbe5b455e5eee502d5701eaca2b1be6d5b5f3fca19276e',
    )
]


def build(opt):
    version = VERSION
    # create particular instance of dataset depending on flags..
    dpath = os.path.join(opt['datapath'], 'fits')

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

    return _get_dpath_by_task_version(dpath, opt)
