#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os
from .download_images import download_images
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/personality_captions/personality_captions.tgz',
        'personality_captions.tgz',
        'e0979d3ac0854395ee74f2c61a6bc467838cc292c3a9a62e891d8230d3a01365',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'personality_captions')
    image_path = os.path.join(opt['datapath'], 'yfcc_images')
    version = '2.0'
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(dpath, version)

    if not build_data.built(image_path, version) and not opt.get('yfcc_path'):
        download_images(opt)
