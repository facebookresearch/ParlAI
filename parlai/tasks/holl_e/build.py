#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        '1u5zzfENGbRYVo-HsyFXZc3sJ9FgDTNx4',
        'raw_train_data.json',
        '7380e41ca8c65084140af997057eb9e8f974e08a19fdb40de73a9f96e4b5bd6d',
        from_google=True,
        zipped=False,
    ),
    DownloadableFile(
        '1nRsAyuVZu7L2f2YcxNbxsT1gZzFnQy-P',
        'raw_test_data.json',
        '3fd2cc672fbae118f3545640fc4c4f45a2e9037c98eebd1e64ad2e0ce5d1fe35',
        from_google=True,
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'holl_e')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
