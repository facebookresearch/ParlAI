# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    with open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        with open(inpath) as f:
            lines = [line.strip('\n') for line in f]
            for i in range(len(lines)):
                use = True
                if dtype == 'train' and (i%20) == 0:
                    use = False
                if dtype == 'valid' and (i%20) != 0:
                    use = False
                if use:
                    xy = lines[i].split('OUT: ')
                    x = xy[0].split('IN: ')[1].rstrip(' ').lstrip(' ')
                    y= xy[1].rstrip(' ').lstrip(' ')
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
        fname = 'scan.tgz'
        url = 'https://s3.amazonaws.com/fair-data/parlai/scan/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        ext = os.path.join('dailymail', 'questions')
        create_fb_format(dpath, 'train', os.path.join(dpath, 'tasks_train_simple.txt'))
        create_fb_format(dpath, 'valid', os.path.join(dpath, 'tasks_train_simple.txt'))
        create_fb_format(dpath, 'test', os.path.join(dpath, 'tasks_test_simple.txt'))

        # Mark the data as built.
        build_data.mark_done(dpath, version)
