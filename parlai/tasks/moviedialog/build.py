#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

URLS = [
    'http://parl.ai/downloads/moviedialog/' + 'moviedialog.tar.gz',
    'http://tinyurl.com/' + 'p6tyohj'
]
FILE_NAMES = ['moviedialog.tar.gz', 'p6tyohj.tgz']
SHA256 = ['9b168d30111f13b9cc50e6a15885adae8f86bc0bb7a124d435c43fd0f7e2a9c7',
'6316a6a5c563bc3c133a4a1e611d8ca638c61582f331c500697d9090efd215bb']

def build(opt):
    dpath = os.path.join(opt['datapath'], 'MovieDialog')
    version = '3'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, [URLS[0]], [FILE_NAMES[0]], [SHA256[0]])

        dpath2 = os.path.join(dpath, 'movie_dialog_dataset', 'task4_reddit')
        build_data.make_dir(dpath2)

        build_data.download_check(dpath2, [URLS[1]], [FILE_NAMES[1]], [SHA256[1]])

        build_data.untar(dpath, FILE_NAMES[0])
        build_data.untar(dpath2, FILE_NAMES[1])

        # remove pipes from task 4 labels, only one label per example
        for root, _subfolder, files in os.walk(os.path.join(dpath2, 'task4_reddit')):
            for f in files:
                if f.endswith('txt'):
                    read_fn = os.path.join(root, f)
                    head = 'task4_reddit_'
                    tail = f[len(head) :]
                    write_fn = os.path.join(root, head + 'pipeless_' + tail)
                    with open(read_fn) as read, open(write_fn, 'w') as write:
                        for line in read:
                            write.write(line.replace('|', ' __PIPE__ ') + '\n')

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
