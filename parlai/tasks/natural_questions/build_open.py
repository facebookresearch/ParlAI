#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
import gzip
import shutil

RESOURCES = [
    DownloadableFile(
        "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-train_gold_info.json.gz",
        'train_with_gold.json.gz',
        'b1a42d917a288ab330ee1ae4387efccbf6abba74d0da14e14a5ac38ff72d9ec8',
        zipped=False,
    ),
    DownloadableFile(
        "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-dev_gold_info.json.gz",
        'valid_with_gold.json.gz',
        '38e6ddeaf3918c99e2be920fd51aa587182a949ead4f8aec200f1999c8231df2',
        zipped=False,
    ),
    DownloadableFile(
        "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-test_gold_info.json.gz",
        'test_with_gold.json.gz',
        'a1588adca9cc1db68ae1cd1507b48de110e016740f77812b2e91d5bb4dc6cf17',
        zipped=False,
    ),
]

VERSION = 1


def build(opt):
    dpath = os.path.join(opt['datapath'], 'NaturalQuestionsOpen')
    version = str(VERSION)

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)
            if ".gz" in downloadable_file.file_name:
                with gzip.open(
                    os.path.join(dpath, downloadable_file.file_name), 'rb'
                ) as fin:
                    with open(
                        os.path.join(dpath, downloadable_file.file_name[:-3]), 'wb'
                    ) as fout:
                        shutil.copyfileobj(fin, fout)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
