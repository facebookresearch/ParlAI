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


def _import_google_cloud_client():
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            'Please install Google Cloud Storage API:\n'
            '\tpip install --upgrade google-cloud-storage\n'
            'Or follow the instruction at '
            'https://cloud.google.com/storage/docs/reference/libraries'
        )
    return storage


def _download_with_cloud_storage_client(dpath, sample: bool = False):
    # Initiating the Cloud Storage Client with anonymous credentials
    stm = _import_google_cloud_client()
    storage_client = stm.Client.create_anonymous_client()

    def _download_blob(blob, target):
        with open(target, 'wb') as fout:
            storage_client.download_blob_to_file(blob, fout)

    def _download_blobs_from_list(blobs_list, target_path):
        for blob in tqdm(blobs_list):
            cloud_storage_fname = blob.name.split('/')[-1]
            downloaded_fname = os.path.join(target_path, cloud_storage_fname)
            _download_blob(blob, downloaded_fname)

    # Creating data storage directories
    for dtype in ('train', 'valid'):
        os.makedirs(os.path.join(dpath, dtype), exist_ok=True)

    # Populating a list of train and valid dataset blobs to download
    train_blobs = []
    valid_blobs = []
    for blob in storage_client.list_blobs('natural_questions'):
        blob_name = blob.name
        if not blob_name.endswith('.gz'):  # Not a zipped file
            continue

        if sample and blob_name.startswith('v1.0/sample'):
            if 'train' in blob_name:
                train_blobs.append(blob)
            else:
                valid_blobs.append(blob)
        elif not sample and blob_name.startswith('v1.0/train'):
            train_blobs.append(blob)
        elif not sample and blob_name.startswith('v1.0/dev'):
            valid_blobs.append(blob)

    # Downloading the blobs to their respective dtype directory
    logging.info('Downloading train data ...')
    _download_blobs_from_list(train_blobs, os.path.join(dpath, 'train'))
    logging.info('Downloading valid data ...')
    _download_blobs_from_list(valid_blobs, os.path.join(dpath, 'valid'))


def _untar_dir_files(dtype_path):
    files = os.listdir(dtype_path)
    for fname in tqdm(files):
        build_data.ungzip(dtype_path, fname)


def _untar_dataset_files(dpath):
    for dtype in ('train', 'valid'):
        unzip_files_path = os.path.join(dpath, dtype)
        logging.info(f'Unzipping {dtype} files at {unzip_files_path}')
        _untar_dir_files(unzip_files_path)


def _move_valid_files_from_dev_to_valid(dpath):
    """
    Files from Google are stored at `nq-dev-##.jsonl.gz` and get untar'd to `nq-
    dev-##.jsonl`.

    The agent expects them to be stored at `nq-valid-00.jsonl`. This moves them over if
    need be.
    """
    valid_path = os.path.join(dpath, 'valid')
    for f in os.listdir(valid_path):
        if "dev" in f:
            new = f.replace('dev', 'valid')
            os.rename(os.path.join(valid_path, f), os.path.join(valid_path, new))


def build(opt, sample: bool = False):
    dpath = os.path.join(opt['datapath'], DATASET_NAME_LOCAL)
    if sample:
        dpath = f"{dpath}_sample"
    version = 'v1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
            logging.info('Removed the existing data (old version).')
        build_data.make_dir(dpath)
        _download_with_cloud_storage_client(dpath, sample)
        _untar_dataset_files(dpath)
        _move_valid_files_from_dev_to_valid(dpath)
        build_data.mark_done(dpath, version_string=version)


def build_sample(opt):
    build(opt, True)
