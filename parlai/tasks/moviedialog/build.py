# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'MovieDialog')
    version = '2'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'moviedialog.tar.gz'
        url = 'http://parl.ai/downloads/moviedialog/' + fname
        build_data.download(url, dpath, fname)

        dpath2 = os.path.join(dpath, 'movie_dialog_dataset', 'task4_reddit')
        build_data.make_dir(dpath2)
        url2 = 'http://tinyurl.com/' + 'p6tyohj'
        build_data.download(url2, dpath2, 'p6tyohj.tgz')

        build_data.untar(dpath, fname)
        build_data.untar(dpath2, 'p6tyohj.tgz')

        # remove pipes from task 4 labels, only one label per example
        for root, _subfolder, files in os.walk(os.path.join(dpath2, 'task4_reddit')):
            for f in files:
                if f.endswith('txt'):
                    read_fn = os.path.join(root, f)
                    write_fn = os.path.join(root, f[:-4] + '_pipeless.txt')
                    with open(read_fn) as read, open(write_fn, 'w') as write:
                        for line in read:
                            write.write(line.replace('|', ' __PIPE__ ') + '\n')

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
