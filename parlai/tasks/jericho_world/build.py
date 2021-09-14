#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from git import Repo
import os

import parlai.core.build_data as build_data
from parlai.core.opt import Opt
import parlai.utils.logging as logging


GH_REPO = 'https://github.com/JerichoWorld/JerichoWorld.git'

DATASET_NAME = 'jericho'
VERSION = '1.0'


def build(opt: Opt) -> None:
    dpath = os.path.join(opt['datapath'], DATASET_NAME)
    if build_data.built(dpath, VERSION):
        logging.debug('Data was already built. Skipping the data building.')
        return

    if os.path.exists(dpath):
        logging.debug(f'Removing old/corrupted data in {dpath}.')
        build_data.remove_dir(dpath)

    logging.info(
        f'[building data: {dpath}]\nThis may take a while but only heppens once.'
    )
    logging.info(f'Cloning Github repo {GH_REPO}')
    temp_path = os.path.join(dpath, "temp")
    Repo.clone_from(GH_REPO, temp_path)
    build_data.untar(temp_path, 'data.zip')

    # Copying the unzipped data files to the dpath
    for dt in ('train', 'test'):
        fname = f'{dt}.json'
        fsource = os.path.join(temp_path, 'data', fname)
        fdest = os.path.join(dpath, fname)
        os.rename(fsource, fdest)

    # Removing unused files from the repository
    build_data.remove_dir(temp_path)

    build_data.mark_done(dpath, VERSION)
