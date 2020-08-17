#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import logging
import os

import parlai.core.build_data as build_data

DATASET_PATH_GS = 'gs://natural_questions/v1.0'
GS_UTIL_NOT_FOUND_MESSAGE = (
    'gsutil is required for downloading the dataset. '
    'Please follow the installation instruction from the following link: '
    'https://cloud.google.com/storage/docs/gsutil_install')

class GSUtilsNotFound(Exception):
    pass

def _check_gsutil_available():
    logging.info('checking wether gsutil is installed')
    stream = os.popen('which gsutil')
    gsutil_path = stream.read().strip()
    if not (gsutil_path and os.path.isfile(gsutil_path)):
        raise GSUtilsNotFound(GS_UTIL_NOT_FOUND_MESSAGE)

def _download_with_gsutil(dpath):
    os.system(f'gsutil -m cp -R gs://natural_questions/v1.0 {dpath}')

def build(opt):
    dpath = os.path.join(opt['datapath'], 'NaturalQuestions')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        _check_gsutil_available()
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        _download_with_gsutil(dpath)
