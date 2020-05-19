#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

ROOT_URL = (
    'https://github.com/google-research-datasets/Taskmaster/raw/master/TM-2-2020/data'
)

RESOURCES = [
    DownloadableFile(
        f'{ROOT_URL}/flights.json',
        'flights.json',
        '86b37b5ae25f530fd18ced78800d30c3b54f7b34bb208ecb51842718f04e760b',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/food-ordering.json',
        'food-ordering.json',
        '0a042e566a816a5d0abebe6f7e8cfd6abaa89729ffc42f433d327df7342b12f8',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/hotels.json',
        'hotels.json',
        '975b0242f1e37ea1ab94ccedd7e0d6ee5831599d5df1f16143e71110d6c6006a',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/movies.json',
        'movies.json',
        '6f67c9a1f04abc111186e5bcfbe3050be01d0737fd6422901402715bc1f3dd0d',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/music.json',
        'music.json',
        'e5db60d6576fa010bef87a70a8b371d293d48cde8524c1d3ed7c3022f079d95d',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/restaurant-search.json',
        'restaurant-search.json',
        'fb9735f89e7ebc7c877f976da4c30391af6a6277991b597c0755564657ff8f47',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/sports.json',
        'sports.json',
        '8191531bfa5a8426b1508c396ab9886a19c7c620b443c436ec10d8d4708d0eac',
        zipped=False,
    ),
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'taskmaster-2')
    # define version if any
    version = "1.0"

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
