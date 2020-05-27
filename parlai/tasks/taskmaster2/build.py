#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

ROOT_URL = 'https://github.com/google-research-datasets/Taskmaster/raw/master/TM-2-2020'

RESOURCES = [
    # raw data files
    DownloadableFile(
        f'{ROOT_URL}/data/flights.json',
        'flights.json',
        '86b37b5ae25f530fd18ced78800d30c3b54f7b34bb208ecb51842718f04e760b',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/data/food-ordering.json',
        'food-ordering.json',
        '0a042e566a816a5d0abebe6f7e8cfd6abaa89729ffc42f433d327df7342b12f8',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/data/hotels.json',
        'hotels.json',
        '975b0242f1e37ea1ab94ccedd7e0d6ee5831599d5df1f16143e71110d6c6006a',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/data/movies.json',
        'movies.json',
        '6f67c9a1f04abc111186e5bcfbe3050be01d0737fd6422901402715bc1f3dd0d',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/data/music.json',
        'music.json',
        'e5db60d6576fa010bef87a70a8b371d293d48cde8524c1d3ed7c3022f079d95d',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/data/restaurant-search.json',
        'restaurant-search.json',
        'fb9735f89e7ebc7c877f976da4c30391af6a6277991b597c0755564657ff8f47',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/data/sports.json',
        'sports.json',
        '8191531bfa5a8426b1508c396ab9886a19c7c620b443c436ec10d8d4708d0eac',
        zipped=False,
    ),
    # ontology data files
    DownloadableFile(
        f'{ROOT_URL}/ontology/flights.json',
        'flights.onto.json',
        '1ebc5c982339d24b2dcf50677883fed65b7fcb95f01edbbd3be6357090893c33',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/ontology/food-ordering.json',
        'food-ordering.onto.json',
        '79c1189c16f0ab937bad558c70a0b9b99358f9ed91ea65ce4af37c4b7d999063',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/ontology/hotels.json',
        'hotels.onto.json',
        '22ae51ba546ee7ca03143097782817c4cdd0de74ac84893eaf40b8254aa866d3',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/ontology/movies.json',
        'movies.onto.json',
        '8403283526bb314e871850b98bb86a7987ef0af6fbbe4fb5a089ee9498584476',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/ontology/music.json',
        'music.onto.json',
        '4bcd6dcf1cdc6bdb717e5fdc08b3472dc3d1f4da8a0f8aee917494d79a7fe338',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/ontology/restaurant-search.json',
        'restaurant-search.onto.json',
        'c9ead7985695b3feba1fb955e8407d806e4095f5459485adc5448ae89989e609',
        zipped=False,
    ),
    DownloadableFile(
        f'{ROOT_URL}/ontology/sports.json',
        'sports.onto.json',
        '52f9bbb86ebd9e2b3916185ad4e3e9b8b77d2164d96bd3b98ad67cbaa653757d',
        zipped=False,
    ),
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'taskmaster-2')
    # define version if any
    version = "1.1"

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
