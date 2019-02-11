#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
import shutil


AQUA_BASE_URL = 'https://github.com/deepmind/AQuA/archive/master.zip'


def build(opt):
    dpath = os.path.join(opt['datapath'], 'AQuA')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        fname = 'aqua.zip'

        build_data.download(AQUA_BASE_URL, dpath, fname)

        # uncompress it
        build_data.untar(dpath, fname)

        base_path = os.path.join(dpath, 'AQuA-master')
        new_path = os.path.join(dpath, 'AQuA')

        shutil.move(base_path, new_path)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
