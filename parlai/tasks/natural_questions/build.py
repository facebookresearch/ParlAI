#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import os
from tqdm import tqdm

import parlai.core.build_data as build_data
import parlai.utils.logging as logging

DATASET_NAME_LOCAL = 'NaturalQuestions'


def _download_with_gsutil(dpath):
    for dt in ('train', 'dev'):
        os.system(f'gsutil -m cp -R gs://natural_questions/v1.0/{dt} {dpath}')

def _untar_dir_files(dtype_path):
    files = os.listdir(dtype_path)
    for fname in tqdm(files):
        build_data.ungzip(dtype_path, fname)

def _untar_dataset_files(dpath):
    for dtype in ('train', 'dev'):
        unzip_files_path = os.path.join(dpath, dtype)
        logging.info(f'Unzipping {dtype} files at {unzip_files_path}')
        _untar_dir_files(unzip_files_path)


def build(opt):
    dpath = os.path.join(opt['datapath'], DATASET_NAME_LOCAL)
    version = 'v1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
            logging.info('Removed the existing data (old version).')
        build_data.make_dir(dpath)
        _download_with_gsutil(dpath)
        _untar_dataset_files(dpath)
        build_data.mark_done(dpath, version_string=version)
