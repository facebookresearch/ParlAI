#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import os

from parlai.core import build_data
from parlai.core.opt import Opt


STYLE_LABELED_DATASETS_VERSION = 'v1.1'


TASK_FOLDER_NAME = 'style_gen'
STYLE_LABELED_DATASETS_RESOURCES = [
    build_data.DownloadableFile(
        f'http://parl.ai/downloads/style_gen/style_labeled_datasets__{STYLE_LABELED_DATASETS_VERSION}.tar.gz',
        f'style_labeled_datasets__{STYLE_LABELED_DATASETS_VERSION}.tar.gz',
        '19995a8957cb3e847d1c0ff18e6ce0c231ed711ae19ebaa624012e1782223445',
    )
]
PERSONALITY_LIST_RESOURCES = [
    build_data.DownloadableFile(
        'http://parl.ai/downloads/style_gen/personality_list.txt',
        'personality_list.txt',
        'f527d9315b9d10f8e65021577a7dc4b1777940cea735588485b1c4b5c8c9032a',
        zipped=False,
    )
]


def get_style_labeled_data_folder(datapath: str) -> str:
    return os.path.join(datapath, TASK_FOLDER_NAME, 'labeled_datasets')


def build_style_labeled_datasets(opt: Opt):
    dpath = get_style_labeled_data_folder(datapath=opt['datapath'])

    if not build_data.built(dpath, version_string=STYLE_LABELED_DATASETS_VERSION):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data
        for downloadable_file in STYLE_LABELED_DATASETS_RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built
        build_data.mark_done(dpath, version_string=STYLE_LABELED_DATASETS_VERSION)


def build_personality_list(opt: Opt):
    dpath = os.path.join(opt['datapath'], TASK_FOLDER_NAME)
    version = 'v1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data
        for downloadable_file in PERSONALITY_LIST_RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built
        build_data.mark_done(dpath, version_string=version)
