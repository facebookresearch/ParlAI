#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'SST')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        url_base = (
            'https://raw.githubusercontent.com/openai/'
            'generating-reviews-discovering-sentiment/master/data/')
        fnames = [
            'train_binary_sent.csv',
            'dev_binary_sent.csv',
            'test_binary_sent.csv'
        ]
        for fname in fnames:
            url = url_base + fname
            build_data.download(url, dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
