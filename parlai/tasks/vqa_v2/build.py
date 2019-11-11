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
        'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
        'v2_Questions_Train_mscoco.zip',
        '05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e',
    ),
    DownloadableFile(
        'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
        'v2_Questions_Val_mscoco.zip',
        'e71f6c5c3e97a51d050f28243e262b28cd0c48d11a6b4632d769d30d3f93222a',
    ),
    DownloadableFile(
        'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip',
        'v2_Questions_Test_mscoco.zip',
        '982e2e687a86514b78ea83af356d151976c5e3fb4168a29ca543610574082ad7',
    ),
    DownloadableFile(
        'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
        'v2_Annotations_Val_mscoco.zip',
        '0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e',
    ),
    DownloadableFile(
        'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
        'v2_Annotations_Train_mscoco.zip',
        'fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853',
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'VQA-v2')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        # An older version exists, so remove these outdated files.
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
