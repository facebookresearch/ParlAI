#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/scan/scan.tgz',
        'scan.tgz',
        '7d6695159fab47ef13a8fadd1f5020d5ab500196e71d5114fd52bc9b7fc8d17f',
    )
]


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    with PathManager.open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        with PathManager.open(inpath) as f:
            lines = [line.strip('\n') for line in f]
            for i in range(len(lines)):
                use = True
                if dtype == 'train' and (i % 20) == 0:
                    use = False
                if dtype == 'valid' and (i % 20) != 0:
                    use = False
                if use:
                    xy = lines[i].split('OUT: ')
                    x = xy[0].split('IN: ')[1].rstrip(' ').lstrip(' ')
                    y = xy[1].rstrip(' ').lstrip(' ')
                    s = '1 ' + x + '\t' + y
                    fout.write(s + '\n\n')


def build(opt):
    version = 'v1.0'
    dpath = os.path.join(opt['datapath'], 'SCAN')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        create_fb_format(dpath, 'train', os.path.join(dpath, 'tasks_train_simple.txt'))
        create_fb_format(dpath, 'valid', os.path.join(dpath, 'tasks_train_simple.txt'))
        create_fb_format(dpath, 'test', os.path.join(dpath, 'tasks_test_simple.txt'))

        # Mark the data as built.
        build_data.mark_done(dpath, version)
