#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import parlai.utils.logging as logging

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/uclanlp/gn_glove/master/wordlist/male_word_file.txt',
        'male_word_file.txt',
        'd431679ce3ef4134647e22cb5fd89e8dbee3f04636f1c7cbae5f28a369acf60f',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/uclanlp/gn_glove/master/wordlist/female_word_file.txt',
        'female_word_file.txt',
        '5f0803f056de3fbc459589bce26272d3c5453112a3a625fb8ee99c0fbbed5b35',
        zipped=False,
    ),
]


def build(datapath):
    version = 'v1.0'
    dpath = os.path.join(datapath, 'genderation_bias')
    if not build_data.built(dpath, version):
        logging.info('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
