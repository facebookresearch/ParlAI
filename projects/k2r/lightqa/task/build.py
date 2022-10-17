#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Download and build the data if it does not exist.
"""

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os
from shutil import copyfile


RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/light_project/k2r/light_dialog_wild_summaryqa2_train.json',
        'lightqa-wild-summaryqa2-train.json',
        '0c618e0736317fbb9a688f82777165675b5967ffc5208041da940a3e3a947d25',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/light_project/k2r/light_dialog_wild_summaryqa2_valid.json',
        'lightqa-wild-summaryqa2-valid.json',
        '3646ff1e6549ec82588caaf7da998ef18df629cacdde43d8ce813df545aabe6c',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/light_project/k2r/light_dialog_wild_summaryqa2_test.json',
        'lightqa-wild-summaryqa2-test.json',
        '70804bd77fe7568326a1e229b3ece578cd1867c3e0e8a14fef23faf4e2032f14',
        zipped=False,
    ),
]


def build(opt):
    version = 'v1.0.0'
    dpath = os.path.join(opt['datapath'], 'lightqa')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            if downloadable_file.url.startswith('/checkpoint'):
                copyfile(
                    downloadable_file.url,
                    os.path.join(dpath, downloadable_file.file_name),
                )
            else:
                downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
