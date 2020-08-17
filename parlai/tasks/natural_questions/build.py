#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import os
from tqdm import tqdm

import parlai.core.build_data as build_data
import parlai.utils.logging as logging

DATASET_PATH_GS = 'gs://natural_questions/v1.0'
DATASET_NAME_LOCAL = 'NaturalQuestions'
GS_UTIL_NOT_FOUND_MESSAGE = (
    'gsutil is required for downloading the dataset. '
    'Please follow the installation instruction from the following link: '
    'https://cloud.google.com/storage/docs/gsutil_install')

class GSUtilsNotFound(Exception):
    pass

def _check_gsutil_available():
    logging.info('checking whether gsutil is installed')
    stream = os.popen('which gsutil')
    gsutil_path = stream.read().strip()
    if not (gsutil_path and os.path.isfile(gsutil_path)):
        raise GSUtilsNotFound(GS_UTIL_NOT_FOUND_MESSAGE)

def _download_with_gsutil(dpath):
    for dt in ('train', 'test'):
        os.system(f'gsutil -m cp -R gs://natural_questions/v1.0/{dt} {dpath}')

def _untar_dir_files(dtype_path):
    files = os.listdir(dtype_path)
    for fname in tqdm(files):
        build_data.ungzip(dtype_path, fname)

def _untar_dataset_files(dpath):
    for dtype in ('train', 'dev'):
        logging.info(f'Unzipping {dtype} files')
        _untar_dir_files(os.path.join(dpath, dtype))


def build(opt):
    dpath = os.path.join(opt['datapath'], DATASET_NAME_LOCAL)
    version = 'v1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        _check_gsutil_available()
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        _download_with_gsutil(dpath)
        _untar_dataset_files(dpath)
        build_data.mark_done(dpath, version_string=version)
