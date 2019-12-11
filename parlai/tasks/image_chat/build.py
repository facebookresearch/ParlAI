#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os
from parlai.tasks.personality_captions.download_images import download_images

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/image_chat/image_chat.tgz',
        'image_chat.tgz',
        'ad733e181de33f1085166bb7af17fcf228504bd48228ed8cc20c5e7a9fa5d259',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'image_chat')
    image_path = os.path.join(opt['datapath'], 'yfcc_images')
    version = '1.0'
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
        download_images(opt, task='image_chat')
