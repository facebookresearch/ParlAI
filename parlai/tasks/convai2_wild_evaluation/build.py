#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import os
import json
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

RESOURCES = [
    DownloadableFile(
        'http://lnsigo.mipt.ru/export/datasets/convai/convai2_wild_evaluation_0.2.tgz',
        'convai2_wild_evaluation_0.2.tgz',
        'd40ff70275c8d1939a8081707edcf4e71072097d18b9998100a1099d23e29801',
    )
]


def make_parlai_format(data: list, dpath: str):
    train_p = 0.6
    valid_p = 0.2
    test_p = 1 - (train_p + valid_p)

    assert train_p > 0
    assert valid_p > 0
    assert test_p > 0

    data_len = len(data)

    first_valid = int(data_len * train_p)
    first_test = int(data_len * (train_p + valid_p))

    data_train = data[:first_valid]
    data_valid = data[first_valid:first_test]
    data_test = data[first_test:]

    data_train_txt = '\n'.join(data_train)
    data_valid_txt = '\n'.join(data_valid)
    data_test_txt = '\n'.join(data_test)

    path_train = os.path.join(dpath, 'train.txt')
    path_valid = os.path.join(dpath, 'valid.txt')
    path_test = os.path.join(dpath, 'test.txt')

    with open(path_train, 'w') as f_train:
        f_train.write(data_train_txt)

    with open(path_valid, 'w') as f_valid:
        f_valid.write(data_valid_txt)

    with open(path_test, 'w') as f_test:
        f_test.write(data_test_txt)


def build(opt):
    version = '0.2'
    dpath = os.path.join(opt['datapath'], 'ConvAI2_wild_evaluation')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        output_fname = 'convai2_wild_evaluation.json'
        output_path = os.path.join(dpath, output_fname)

        with open(output_path, 'r') as data_f:
            data = json.load(data_f)

        make_parlai_format(data, dpath)
        os.remove(output_path)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
