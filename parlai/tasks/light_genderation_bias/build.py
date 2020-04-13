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


RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/genderation_bias/genderation_bias.tgz',
        'genderation_bias.tgz',
        '9a0252c6bb778757ac60dee9df23a169192f4a853ceb2b530af2343abeb1498a',
    )
]


def build(opt):
    version = 'v1.0'
    dpath = os.path.join(opt['datapath'], 'light_genderation_bias')

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
