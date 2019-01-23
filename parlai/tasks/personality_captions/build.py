# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os
from .download_images import download_images


def build(opt):
    dpath = os.path.join(opt['datapath'], 'personality_captions')
    image_path = os.path.join(opt['datapath'], 'yfcc_images')
    fname = 'personality_captions.tgz'
    version = '2.0'
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        url = 'http://parl.ai/downloads/personality_captions/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)
        build_data.mark_done(dpath, version)

    if not build_data.built(image_path, version) and not opt.get('yfcc_path'):
        download_images(opt)
