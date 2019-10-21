# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os
from parlai.tasks.personality_captions.download_images import download_images

URLS = ['http://parl.ai/downloads/image_chat/image_chat.tgz']
FILE_NAMES = ['image_chat.tgz']
SHA256 = ['ad733e181de33f1085166bb7af17fcf228504bd48228ed8cc20c5e7a9fa5d259']

def build(opt):
    dpath = os.path.join(opt['datapath'], 'image_chat')
    image_path = os.path.join(opt['datapath'], 'yfcc_images')
    fname = 'image_chat.tgz'
    version = '1.0'
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)
        for zipfile in FILE_NAMES:
            build_data.untar(dpath, zipfile)
        build_data.mark_done(dpath, version)

    if not build_data.built(image_path, version) and not opt.get('yfcc_path'):
        download_images(opt, task='image_chat')
