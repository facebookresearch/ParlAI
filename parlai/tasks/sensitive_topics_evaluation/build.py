#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/sensitive_topics_evaluation/data_valid.jsonl',
        'data_valid.jsonl',
        'df3a71da78bd231402237fded6df530c80f91814f03a2c3e0581be14fe24633d',
        zipped=False,
    )
]


def build(opt):
    version = 'v1.0'
    dpath = os.path.join(opt['datapath'], 'sensitive_topics_evaluation')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

    return os.path.join(dpath, 'data_valid.jsonl')
