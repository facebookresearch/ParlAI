#!/usr/bin/env python3

# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/deepmipt/turing-data/master/data_1501534800.tar.gz',
        'data_1501534800.tar.gz',
        'f1a4e7ee7264220cef6bf067b77d6f501023877643e77516c7acd66fbcdf0aaf',
    )
]


def build(opt):
    data_path = os.path.join(opt['datapath'], 'DialogueQE')
    version = '1501534800'

    if not build_data.built(data_path, version_string=version):
        print('[building data: ' + data_path + ']')

        if build_data.built(data_path):
            build_data.remove_dir(data_path)
        build_data.make_dir(data_path)

        for downloadable_file in RESOURCES:
            downloadable_file.download_file(data_path)

        os.rename(
            os.path.join(data_path, 'data_train_' + version + '.json'),
            os.path.join(data_path, 'train.json'),
        )
        os.rename(
            os.path.join(data_path, 'data_test_' + version + '.json'),
            os.path.join(data_path, 'test.json'),
        )

        build_data.mark_done(data_path, version_string=version)
