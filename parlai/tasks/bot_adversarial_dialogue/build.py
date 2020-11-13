#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import os

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile
from parlai.core.opt import Opt


BOT_ADVERSARIAL_DIALOGUE_DATASETS_VERSION = 'v0.2'
HUMAN_SAFETY_EVAL_TESTSET_VERSION = 'v0.1'

TASK_FOLDER_NAME = 'bot_adversarial_dialogue'

BOT_ADVERSARIAL_DIALOGUE_DATASETS_RESOURCES = [
    DownloadableFile(
        f'http://parl.ai/downloads/bot_adversarial_dialogue/dialogue_datasets_{BOT_ADVERSARIAL_DIALOGUE_DATASETS_VERSION}.tar.gz',
        f'dialogue_datasets_{BOT_ADVERSARIAL_DIALOGUE_DATASETS_VERSION}.tar.gz',
        '2178b022fac154ddd9b570f6386abc4cd3e7ceb4476f0bebfbce5941424461eb',
    )
]
HUMAN_SAFETY_EVAL_TESTSET_RESOURCES = [
    build_data.DownloadableFile(
        f'http://parl.ai/downloads/bot_adversarial_dialogue/human_safety_eval_{HUMAN_SAFETY_EVAL_TESTSET_VERSION}.tar.gz',
        f'human_safety_eval_{HUMAN_SAFETY_EVAL_TESTSET_VERSION}.tar.gz',
        'b8b351c3e5eefcd54fdd73cd6a04847cd1eeb9106fc53b92a87e2a4c7537a7b2',
    )
]


def get_adversarial_dialogue_folder(datapath: str) -> str:
    return os.path.join(datapath, TASK_FOLDER_NAME, 'dialogue_datasets')


def get_human_safety_eval_folder(datapath: str) -> str:
    return os.path.join(datapath, TASK_FOLDER_NAME, 'human_eval')


def build_dialogue_datasets(opt: Opt):
    dpath = get_adversarial_dialogue_folder(opt['datapath'])
    version = BOT_ADVERSARIAL_DIALOGUE_DATASETS_VERSION
    downloadable_files = BOT_ADVERSARIAL_DIALOGUE_DATASETS_RESOURCES

    return build_data_from_path(dpath, version, downloadable_files)


def build_human_safety_eval_dataset(opt: Opt):
    dpath = get_human_safety_eval_folder(opt['datapath'])
    version = HUMAN_SAFETY_EVAL_TESTSET_VERSION
    downloadable_files = HUMAN_SAFETY_EVAL_TESTSET_RESOURCES

    return build_data_from_path(dpath, version, downloadable_files)


def build_data_from_path(dpath, version, downloadable_files):
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data
        for downloadable_file in downloadable_files:
            downloadable_file.download_file(dpath)

        # Mark the data as built
        build_data.mark_done(dpath, version_string=version)

    return dpath, version
